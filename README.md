<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo/bgpgen-logo-dark.svg">
    <img src="docs/logo/bgpgen-logo.svg" alt="BGPgen" width="360">
  </picture>
</p>

<p align="center">
  <strong>Sample diverse SPARQL basic graph pattern queries with exact cardinalities from an RDF graph fast.</strong>
</p>

<p align="center">
  <img alt="C++20" src="https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white">
  <img alt="Python 3.9 or newer" src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white">
  <img alt="Platforms: Linux, macOS" src="https://img.shields.io/badge/platforms-Linux%20%7C%20macOS-555555">
  <a href="https://opensource.org/licenses/Apache-2.0"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-1D4ED8"></a>
  <a href="https://doi.org/10.5281/zenodo.19729261"><img alt="DOI" src="https://zenodo.org/badge/738566926.svg"></a>
</p>

BGPgen takes an RDF graph and a
query shape and returns queries that have at least one result on that graph,
each with its exact result size. The queries are basic graph patterns (BGPs):
sets of triple patterns, as in the `WHERE` clause of a SPARQL query. Such
workloads are used to train and to test cardinality estimators, query
optimisers and knowledge graph embeddings.

The engine `bgpgen` is written in C++20 and needs no other library. It loads
the graph into memory, so no SPARQL endpoint is needed. It comes as a
command-line tool, a C++ library and a Python package.

## Highlights

- **Any connected shape.** Stars and paths by size, every other shape by a
  template string such as `"?x B ?y, ?y B ?z, ?z B ?x"` (a triangle).
- **Exact cardinalities.** Every query comes with its `COUNT(*)` on the
  graph, computed in memory.
- **No duplicates.** Two queries that differ only in the names of their
  variables count as the same query. One of them is kept.
- **Control over the workload.** Options spread the queries over the
  predicates of the graph, or fill ranges of the result size evenly.
- **Fast.** 1000 path queries with their cardinalities on a graph with 21
  million triples take about 0.2 s after loading (see [Performance](#performance)).
- **Repeatable.** The same seed gives the same workload (see
  [Scope and limits](#scope-and-limits) for the one condition).

## Quick start

Download the command-line tool and the example graph, then sample triangle
queries (on macOS with Apple Silicon use `bgpgen-macos-arm64`):

```sh
curl -L -o bgpgen https://github.com/DE-TUM/rdf-subgraph-sampler/releases/latest/download/bgpgen-linux-x86_64
chmod +x bgpgen
curl -L -O https://raw.githubusercontent.com/DE-TUM/rdf-subgraph-sampler/main/examples/toy.nt
./bgpgen --data toy.nt --shape complex --template "?x B ?y, ?y B ?z, ?z B ?x"
```

The last command prints what it did and writes a JSON file to the current
folder:

```
49 triples (49 distinct), 8 predicates, 30 terms (25 iris, 0 bnodes, 5 literals)
complex: 2/1000 unique queries (8 duplicates skipped, 0 reshuffles) - stopped: diminishing returns (<5 new queries in a pass) -> ./toy_complex_2026-11-26_11-12-55_3.json
```

The toy graph has two different triangles, so the run ends with two queries.
One of them:

```json
{
  "query": "SELECT * WHERE { ?x <http://example.org/worksFor> ?y . ?y <http://example.org/funded> ?z . ?z <http://example.org/author> ?x }",
  "triples": [["?x", "<http://example.org/worksFor>", "?y"],
              ["?y", "<http://example.org/funded>", "?z"],
              ["?z", "<http://example.org/author>", "?x"]],
  "x": ["http://example.org/worksFor", "http://example.org/funded", "http://example.org/author"],
  "y": 6
}
```

## Installation

### Prebuilt files

Every [release](https://github.com/DE-TUM/rdf-subgraph-sampler/releases)
comes with ready-made files for Linux (x86_64) and macOS (Apple Silicon).
Nothing has to be compiled.

**Command-line tool.** One file, no installation:

```sh
# Linux
curl -L -o bgpgen https://github.com/DE-TUM/rdf-subgraph-sampler/releases/latest/download/bgpgen-linux-x86_64
# macOS
curl -L -o bgpgen https://github.com/DE-TUM/rdf-subgraph-sampler/releases/latest/download/bgpgen-macos-arm64

chmod +x bgpgen
./bgpgen --help
```

Move the file to a folder on your `PATH` (for example `~/.local/bin`) to
call it as `bgpgen` from anywhere. The rest of this README assumes that. If
you downloaded the file with a browser on macOS, remove the download mark
first: `xattr -d com.apple.quarantine bgpgen`.

**Python package.** Wheels exist for Python 3.9 to 3.13. Download the wheel
for your Python version and platform from the
[release page](https://github.com/DE-TUM/rdf-subgraph-sampler/releases/latest)
and install the file:

```sh
pip install bgpgen-2.0.0-cp312-cp312-macosx_11_0_arm64.whl   # the file you downloaded
```

In the file name, `cp312` stands for Python 3.12, and `manylinux` or
`macosx` for the platform. The wheel contains the module `bgpgen` (see
[Python bindings](#python-bindings)) and the command-line tool.

### Build from source

On other platforms, or to change the code, build it yourself. Requirements:
a C++20 compiler and CMake 3.20 or newer. zlib is optional and adds support
for `.nt.gz` input.

```sh
git clone https://github.com/DE-TUM/rdf-subgraph-sampler.git
cd rdf-subgraph-sampler
cmake -S cpp -B build          # Release build by default
cmake --build build -j
build/bgpgen_tests             # optional: run the test suite
```

The binary is `build/bgpgen`. The Python package is built with
`pip install .` in the same folder.

| Platform | Tested with |
|---|---|
| Linux | Ubuntu 24.04, GCC 13.3, CMake 3.28, Python 3.12 |
| macOS | Apple clang 21, CMake 4.3, Python 3.9 and 3.12 |

The build uses `-march=native` by default. Pass `-DBGPGEN_NATIVE=OFF` to
`cmake` if the binary has to run on a different machine than the one that
built it. The prebuilt files and `pip install .` are built without it; for
the fastest Python package on your own machine use
`pip install . -C cmake.define.BGPGEN_NATIVE=ON`. Windows is not supported.

## Usage

```
bgpgen --data graph.nt[.gz] --shape star|path|complex [--size N] [--template "..."] [options]
```

`bgpgen --help` lists all options. Without `--shape` the tool loads the graph
and prints its size.

### Input

The graph has to be in N-Triples (`.nt` or `.nt.gz`). Other formats can be
converted first, for example with Apache Jena
(`riot --output=nt data.ttl > data.nt`) or Raptor
(`rapper -i turtle -o ntriples data.ttl > data.nt`).

### Shapes

| Shape | Command | Meaning |
|---|---|---|
| Star | `--shape star --size 4` | 4 triple patterns that share their subject |
| Path | `--shape path --size 3` | a chain of 3 triple patterns |
| Any other | `--shape complex --template "..."` | the shape given by the template |

`--size` can be given several times (`--size 2 --size 3 --size 5`). The tool
then writes one file per size.

### Templates

A template is a list of triple patterns, separated by commas. Each pattern
has three parts: subject, predicate, object.

| Position | Token | In the sampled query |
|---|---|---|
| Subject, object | `?x` | the variable `?x` |
| | `x` | a constant: the term that the graph has at this place |
| | `?x/B` | a constant with probability `--p-node`, else the variable `?x` |
| Predicate | `B` | a constant: the predicate that the graph has at this place |
| | `?` | a variable |
| | `?/B` | a constant with probability `--p-edge`, else a variable |

A name that occurs in two patterns stands for the same node, so shared names
define the shape. Some examples:

```sh
# Triangle
--template "?x B ?y, ?y B ?z, ?z B ?x"
# Diamond
--template "?a B ?b, ?a B ?c, ?b B ?d, ?c B ?d"
# Tree: a root with two children, one grandchild below each
--template "?p B ?a, ?p B ?b, ?a B ?c, ?b B ?d"
# Path of two triple patterns; the start node is a constant in half of the queries
--template "?x/B B ?y, ?y B ?z" --p-node 0.5
```

### The most used options

| Option | Default | Meaning |
|---|---|---|
| `--num N` | 1000 | number of queries per size |
| `--seed N` | 42 | seed of the random number generator |
| `--output-dir DIR` | `.` | folder for the output files (it has to exist) |
| `--dataset NAME` | name of the data file | first part of the output file names |
| `--no-cardinality` | off | do not count the results (`y` is then -1) |
| `--card-budget N` | 50000000 | steps the counter may take per query; beyond that `y` is -1 |
| `-j N` | all cores | threads for loading and counting |
| `--timing` | off | print the time of every phase |

The options that change which queries are sampled are described in the
[tutorial](#tutorial) and, in full, in [cpp/README.md](cpp/README.md).

### Output

One JSON file per run and size, named
`{dataset}_{stars|path|complex}_{timestamp}_{size}.json`
(path files end in `_dfs.json`). It holds a list of records:

| Key | Content |
|---|---|
| `query` | the query as a SPARQL string |
| `triples` | the triple patterns as lists of three strings |
| `x` | the constants of the query, without angle brackets |
| `y` | the number of results (`COUNT(*)`), or -1 if it was not counted |

Records of star queries have some more keys (`source_subject`, `star_size`,
`predicates_used`, and others). The count `y` is the number a SPARQL endpoint
returns for `SELECT (COUNT(*) AS ?c) WHERE { ... }` on the same graph.

## Tutorial

This walks from a graph file to a workload. The commands use the toy graph
in [`examples/toy.nt`](examples/toy.nt) (49 triples about papers, authors and
conferences). Replace it by your own `.nt` file. `bgpgen` is the tool from the
[installation](#installation); after a build from source it is `build/bgpgen`.

**1. Look at the graph.** Without a shape, the tool loads the graph and
prints its size:

```sh
bgpgen --data examples/toy.nt
```

```
49 triples (49 distinct), 8 predicates, 30 terms (25 iris, 0 bnodes, 5 literals)
```

**2. Sample stars and paths.**

```sh
mkdir -p workload
bgpgen --data examples/toy.nt --shape star --size 2 --num 5 --output-dir workload
bgpgen --data examples/toy.nt --shape path --size 2 --num 5 --output-dir workload
```

Each run reports how many queries it found and where it wrote them:

```
size 2: 5/5 unique queries (0 duplicates skipped, 0 reshuffles) -> workload/toy_stars_..._2.json
```

By default a star query has between 0 and half of its objects as constants,
and a path query has its end nodes as constants with probability 0.3
(`--p-start-end`).

**3. Sample your own shape.** Write the shape as a template:

```sh
bgpgen --data examples/toy.nt --shape complex --output-dir workload \
       --template "?paper B ?person, ?person B ?org, ?org B ?paper"
```

If the graph has fewer different queries of a shape than `--num`, the run
stops by itself and says why (`stopped: ...`).

**4. Large workloads.** On a large graph, these settings find new queries
faster and spread them over more predicates:

```sh
bgpgen --data graph.nt --shape complex --num 5000 \
       --template "?p B ?a, ?p B ?b, ?a B ?c, ?b B ?d" \
       --attempt-budget 100 --novelty 1 --skip-known --max-known 1 \
       --per-seed auto --retire-seeds
```

Add `--seed-groups` to visit start nodes with rare combinations of predicates
as often as common ones.

**5. A workload with small and large results.** Most sampled queries have few
results. To get the same number of queries in each range of the result size:

```sh
bgpgen --data graph.nt --shape path --size 3 --num 1000 \
       --card-buckets 5 --card-max 100000
```

This asks for 200 queries in each of 5 ranges between 1 and 100000 results,
equally wide on a log scale. The tool reports how full each range got (here
on a graph with 242,256 triples, where the last range has too few queries):

```
queries per range: 200 200 200 200 171; 3424 counted in 0.093 s, dropped: 2394 range full, 59 outside
```

**6. Read the workload.** The output is plain JSON (or use the [Python bindings](#python-bindings)):

```python
import json

with open("workload/toy_path_2026-11-26_11-09-22_2_dfs.json") as f:
    workload = json.load(f)

for record in workload:
    print(record["y"], record["query"])
```

## Python bindings

Install the wheel from the release page (see [Installation](#installation)),
or build the package from source with `pip install .`. It needs Python 3.9
or newer.

```python
import bgpgen

graph = bgpgen.Graph("examples/toy.nt")        # loads the graph once
print(graph.triples, graph.predicates)         # 49 8

queries = graph.sample("?x B ?y, ?y B ?z, ?z B ?x")   # any template
paths = graph.sample_paths(3, num=1000, seed=7)
stars = graph.sample_stars(4, num=1000)

for q in queries:
    print(q["y"], q["query"])
print(graph.last_report["stop_reason"])        # why a run ended early, or ""
```

Each call returns a list of dicts. They are the records of the
[JSON output](#output), so `json.dump(queries, f)` writes the file that the
command-line tool writes. For the same options and seed, the module and the
tool give the same queries.

The options are the options of the command line as keyword arguments, with
`_` in place of `-`:

```python
queries = graph.sample("?p B ?a, ?p B ?b, ?a B ?c, ?b B ?d", num=5000,
                       attempt_budget=100, novelty=1, skip_known=True, max_known=1,
                       per_seed="auto", retire_seeds=True)
paths = graph.sample_paths(3, num=1000, card_buckets=5, card_max=100000)
```

A flag that starts with `--no-` becomes a keyword set to `False`
(`cardinality=False`, `auto_prune=False`). `bgpgen.DEFAULTS` lists all
options with their defaults. An option that a call does not know raises a
`TypeError`. A file that cannot be read or a wrong template raises
`bgpgen.BgpgenError`. Loading, sampling and counting release the GIL.

`pip` also installs the command-line tool `bgpgen`.

### The Python sampler of version 1

The first implementation of the sampler was written in pure Python. It is
still in this repository (`sampler.py`, `batch_sampler.py`, `samplers/`) and
is described in [docs/python-sampler.md](docs/python-sampler.md). Use it if
you have to sample through a SPARQL endpoint, because the graph does not fit
in memory. In every other case `bgpgen` is the better choice: it is faster by
orders of magnitude and counts the results itself.

## Performance

Measured on an Apple Silicon laptop with all cores. The times of the C++
engine include the exact cardinalities. The Python sampler needs an endpoint
for them (1 to 3 s per query), so its times are without cardinalities.

| | LUBM (1.0M triples) | Wikidata subset (21.4M triples) |
|---|---|---|
| Load and index | 0.35 s | 14.2 s |
| 1000 star queries of size 4, with counts | 0.13 s | 1.7 s |
| 1000 path queries of size 3, with counts | 0.02 s | 0.2 s |
| Python sampler: load and 1000 stars of size 4 | | 64.4 s |
| Python sampler: load and 1000 paths of size 3 | | 103.8 s |

## Scope and limits

- The queries are basic graph patterns. There is no `FILTER`, `OPTIONAL`,
  `UNION` or property path.
- Every query has at least one result on the graph. Queries with an empty
  result are not produced.
- The input format is N-Triples.
- The graph has to fit in memory.
- In path and complex queries, all nodes are IRIs. Literals appear only as
  objects of star queries.
- A template has to be connected.
- The same seed gives the same workload on the same platform. The order of
  random draws depends on the C++ standard library, so a Linux build and a
  macOS build give different workloads for the same seed.
- Counting stops after `--card-budget` steps. Such a query has `y = -1`.

## Repository

```
cpp/                the C++ engine
  include/, src/    library: parser, indexes, search, duplicate check, counter
  cli/              the bgpgen binary
  bindings/python/  the Python module
  tests/            test suite (doctest)
  bench/            drivers for the experiments on the sampling options
  README.md         all options of the engine and how they work
python/             the Python package and its tests
pyproject.toml      build settings of the Python package
examples/toy.nt     the graph used in this README
docs/               logo, documentation of the Python sampler of version 1
sampler.py, batch_sampler.py, samplers/   the Python sampler of version 1
```

## Tests

In a source checkout, after the build:

```sh
build/bgpgen_tests               # C++ engine: 44 test cases
pip install . pytest
pytest python/tests              # Python bindings: 12 tests
python test_canonicalization.py  # duplicate check of the Python sampler of version 1
```

The tests of the bindings also compare their output with the output of the
command-line tool for the same seed. Every push runs these tests on Linux
and macOS, and builds the wheels and binaries
([workflows](.github/workflows)).

The counter is tested against a brute-force counter on random graphs, and
was compared with a SPARQL endpoint on Wikidata (30 of 30 queries equal).

## Citation

If you use this software, please cite it:

```bibtex
@software{schwabe_rdf_subgraph_sampler_2026,
  author    = {Schwabe, Tim and Acosta, Maribel},
  title     = {{rdf-subgraph-sampler}: Sampling Subgraph Templates
               and Cardinalities from RDF Knowledge Graphs},
  year      = {2026},
  version   = {2.0.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19729261},
  url       = {https://github.com/DE-TUM/rdf-subgraph-sampler}
}
```

The DOI [10.5281/zenodo.19729261](https://doi.org/10.5281/zenodo.19729261)
stands for all versions and resolves to the latest one. Every release also
has a DOI of its own on Zenodo.

## Maintenance and contact

**Maintainer:** Tim Schwabe, Technical University of Munich
([Data Engineering Group](https://www.cs.cit.tum.de/cde/homepage/), led by Prof. Maribel Acosta).

Questions and bug reports: please open an
[issue](https://github.com/DE-TUM/rdf-subgraph-sampler/issues) or write to
`tim.schwabe@tum.de`.

## License

Apache License 2.0, see [LICENSE](LICENSE). The test framework doctest in
`cpp/third_party/` is under the MIT license.
