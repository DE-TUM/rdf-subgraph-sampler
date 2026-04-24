# rdf-subgraph-sampler

### Sampling Subgraph Templates and Cardinalities from RDF Knowledge Graphs

[![DOI](https://zenodo.org/badge/738566926.svg)](https://doi.org/10.5281/zenodo.19729261)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)



Sampling approach to obtain conjunctive queries from RDF knowledge graphs and their cardinality. The tool supports both endpoint-based and in-memory generation methods for extracting star, path, and arbitrary complex query patterns.

**Maintainer:** Tim Schwabe — Technical University of Munich
(*[Data Engineering Group](https://www.cs.cit.tum.de/cde/homepage/)*, led by Prof. Maribel Acosta)

Contact: `tim.schwabe@tum.de` (or open an issue).

## Installation 
* Clone the repository
* Install the required Python libraries: `tqdm` and `requests`
* For endpoint-based generation: Load the RDF knowledge graph in a SPARQL endpoint
* For in-memory generation: Have your RDF file available locally (supports .nt format) (but you also need to load it if you want to calculate cardinalities)
* We recommend using in-memory generation whenever feasible, it is significantly faster

## Usage 

### Single Configuration Generation
For generating queries with a single configuration:

1. Create `sampler_config.py`
2. Edit the configuration in `sampler_config.py` (see below)
3. Run: `python sampler.py`

### Batch Generation
For generating multiple query configurations in one run:
1. Create `batch_sampler_config.py` 
2. Edit the configuration in `batch_sampler_config.py` 
3. Run: `python batch_sampler.py`

## Configuration Files

The tool uses separate configuration files to manage settings:

### `sampler_config.py`
Configuration file for single query generation (`sampler.py`). Contains all standard parameters including query size, number of queries, and sampling methods.

**Example:**
```python
# General configuration
ENDPOINT = "http://localhost:8890/sparql"  # URL of the SPARQL endpoint where the KG is hosted (required)
QUERIES = 10  # Maximum number of queries to generate
SIZE = 3  # Number of triple patterns in the queries
SHAPE = "path"  # Shape of subgraphs to generate: "path", "star", or "complex"
DATASET = "LUBM_TEST"  # Dataset name (used for output file)
USE_CACHE = True  # Whether to use cached search structures for in memory generation (created on first run for new rdf file)
IN_MEMORY = True  # Whether to use in-memory search and generation instead of sparql endpoint

RDF_FILE_PATH = "/../../..nt"  # Path to RDF file to use

GET_CARDINALITY = True  # Whether to compute cardinality for generated queries in-memory

# Star-specific configuration
# How many objects to keep instantiated - note: number is chosen uniformly at random between min and max
MIN_OBJECTS_INSTANTIATED = 0  # Minimum number of objects to keep instantiated
MAX_OBJECTS_INSTANTIATED = 5  # Maximum number of objects to keep instantiated
P_PREDICATE = 1.0  # Probability of instantiating predicates (1.0 = always instantiate, 0.0 = never instantiate)

# Path-specific configuration
SAMPLING_METHOD = "dfs"  # Sampling method in-memory: "dfs", "bfs" or "random_walk"
P_EDGE = 1.0  # Probability of instantiating a predicate in path queries (1.0 = always instantiate, 0.0 = never instantiate)
P_NODE = 0.3  # Probability of instantiating an intermediate node in path queries
P_START_END = 0.3  # Probability of instantiating start/end node (non-seed) in path queries

# Complex-specific configuration
QUERY_SHAPE = None  # Shape string, e.g. "?a B ?b, ?b B ?c, ?c B ?a"

DEDUP_METHOD = "hash"  # Deduplication: "hash" (fast, multiset) or "wl" (precise, isomorphism)
```

### `batch_sampler_config.py`
Configuration file for batch query generation (`batch_sampler.py`). Includes all standard parameters plus batch-specific configurations.

**Example:**
```python
# Configuration file for batch_sampler.py

# General configuration
ENDPOINT = "http://localhost:8890/sparql"  # URL of the SPARQL endpoint where the KG is hosted (required)
SHAPE = "complex"  # Shape of subgraphs to generate: "path", "star", or "complex"
DATASET = "LUBM_TEST"  # Dataset name (used for output file)
USE_CACHE = True  # Whether to use cached search structures for in memory generation
IN_MEMORY = True  # Whether to use in-memory search and generation instead of sparql endpoint

RDF_FILE_PATH = "....nt"  # Path to RDF file to use

GET_CARDINALITY = True  # Whether to compute cardinality for generated queries in-memory

# Star-specific configuration
# NOTE: MIN_OBJECTS_INSTANTIATED and MAX_OBJECTS_INSTANTIATED are calculated dynamically
# based on query size (min=0, max=ceil(0.5 * size)) in batch sampling
P_PREDICATE = 1.0  # Probability of instantiating predicates

# Path-specific configuration
SAMPLING_METHOD = "dfs"  # Sampling method: "dfs", "bfs" or "random_walk"
P_EDGE = 1.0  # Probability of instantiating a predicate
P_NODE = 0.3  # Probability of instantiating an intermediate node
P_START_END = 0.3  # Probability of instantiating start/end node

# Complex-specific configuration
QUERY_SHAPE = "?r B ?a, ?r B ?b, ?a B ?c, ?a B ?d, ?b B ?e, ?b B ?f"  # Binary tree

DEDUP_METHOD = "hash"  # Deduplication: "hash" (fast, multiset) or "wl" (precise, isomorphism)

# ====== BATCH CONFIGURATION ======
# Specify multiple query sizes and how many queries to generate for each size
# Format: [(size, number_of_queries), (size, number_of_queries), ...]
QUERY_CONFIGURATIONS = [
    (3, 10),    # Generate 10 queries of size 3 (max objects: ceil(0.5*3) = 2)
    (5, 15),    # Generate 15 queries of size 5 (max objects: ceil(0.5*5) = 3)
    (7, 20),    # Generate 20 queries of size 7 (max objects: ceil(0.5*7) = 4)
    (10, 25),   # Generate 25 queries of size 10 (max objects: ceil(0.5*10) = 5)
] 
```

## Configuration Options

### General Parameters
* `ENDPOINT`: URL of the SPARQL endpoint (required for endpoint-based generation and cardinality computation)
* `QUERIES`: Number of queries to generate (single mode only)
* `SIZE`: Number of triple patterns in the queries (single mode only)
* `SHAPE`: Shape of subgraphs to generate - `"path"`, `"star"`, or `"complex"`
* `DATASET`: Dataset name used for output file naming
* `USE_CACHE`: Whether to use cached search structures for faster generation (default: True) - only for in-memory Generation
* `IN_MEMORY`: Whether to use in-memory generation instead of SPARQL endpoint queries (default: True)
* `RDF_FILE_PATH`: Path to the RDF file for in-memory generation (required when `IN_MEMORY=True`)
* `GET_CARDINALITY`: Whether to compute cardinality for generated queries if in-memory(default: True)

### Deduplication
* `DEDUP_METHOD`: Method used to detect duplicate query patterns during generation - `"hash"` or `"wl"` (default: `"hash"`)
  * **`"hash"`** — Hashes the sorted multiset of `(predicate, bound_object_or_?)` pairs. Very fast (~2 µs/query). Ignores join structure, so it may reject queries that have the same predicate/object multiset but different variable connectivity. This means it is **stricter** than necessary: it never keeps duplicates, but may discard some unique queries. Accurate for star queries (where the multiset fully determines the structure).
  * **`"wl"`** — Computes a canonical form using Weisfeiler-Leman color refinement with individualize-and-refine. Still fast (~60 µs/query) but ~30x slower than hash. Respects the full join structure, so two queries are marked as duplicates only if they are truly isomorphic (identical up to variable renaming). Accurate for any query shape (stars, paths, cycles, diamonds, trees, etc.).

  For star queries both methods are equivalent. For paths and complex shapes, `"wl"` will typically retain more unique queries (e.g., ~12% more on path queries in our tests). Use `"hash"` when speed is critical and a small loss of unique queries is acceptable; use `"wl"` when maximizing the number of unique query patterns matters.

### Star-Specific Parameters
* `MIN_OBJECTS_INSTANTIATED`: Minimum number of objects to keep instantiated (default: 0)
* `MAX_OBJECTS_INSTANTIATED`: Maximum number of objects to keep instantiated (default: 5)
* `P_PREDICATE`: Probability of instantiating predicates (1.0 = always, 0.0 = never, default: 1.0)

**Note for Batch Mode:** In batch sampling, `MIN_OBJECTS_INSTANTIATED` and `MAX_OBJECTS_INSTANTIATED` are calculated dynamically based on query size:
- `MIN_OBJECTS_INSTANTIATED = 0` (always)
- `MAX_OBJECTS_INSTANTIATED = ceil(0.5 * query_size)`

### Path-Specific Parameters
* `SAMPLING_METHOD`: Sampling method for path generation - `"dfs"`, `"bfs"`, or `"random_walk"` (default: "dfs")
* `P_EDGE`: Probability of instantiating a predicate in path queries (1.0 = always instantiate, 0.0 = never instantiate, default: 1.0)
* `P_NODE`: Probability of instantiating an intermediate node in path queries (default: 0.3)
* `P_START_END`: Probability of instantiating start/end node (non-seed) in path queries (default: 0.3)

### Complex-Specific Parameters
* `QUERY_SHAPE`: A shape string defining the query topology. Each comma-separated group is a triple pattern: `<subject> <predicate_binding> <object>`.

**Node tokens:**
* `?name` - always a variable in the generated query
* `name` - always bound (instantiated with the matched URI)
* `?name/B` - probabilistically bound (controlled by `P_NODE`)

**Predicate tokens:**
* `B` - always bound
* `?` - always a variable
* `?/B` - probabilistically bound (controlled by `P_EDGE`)

Shared variable names across triples define the query topology. For example, `?o` appearing as the object in one triple and the subject in another creates a join.

**Example shapes:**

```
# Binary tree depth 3 (6 triples):
#          ?r
#         / \
#       ?a   ?b
#       /\   /\
#     ?c ?d ?e ?f
QUERY_SHAPE = "?r B ?a, ?r B ?b, ?a B ?c, ?a B ?d, ?b B ?e, ?b B ?f"

# Flower — path with stars at each node (7 triples):
QUERY_SHAPE = "?a B ?l1, ?a B ?l2, ?a B ?b, ?b B ?m1, ?b B ?c, ?c B ?r1, ?c B ?r2"

# Cycle / triangle (3 triples):
QUERY_SHAPE = "?a B ?b, ?b B ?c, ?c B ?a"

# Diamond (4 triples):
QUERY_SHAPE = "?a B ?b, ?a B ?c, ?b B ?d, ?c B ?d"

# Snowflake — star center with branching arms (8 triples):
QUERY_SHAPE = "?center B ?a, ?center B ?b, ?center B ?c, ?center B ?d, ?a B ?l1, ?a B ?l2, ?b B ?r1, ?b B ?r2"
```

The complex generator uses DFS-based subgraph matching with both forward and reverse adjacency lookups, so it supports shapes where a node appears as both subject and object (e.g., cycles, diamonds).

### Batch-Specific Parameters
* `QUERY_CONFIGURATIONS`: List of `(size, number_of_queries)` tuples defining the batch configurations. For `SHAPE = "complex"`, the size is derived from `QUERY_SHAPE` and the first element is ignored.

### Bulk Shape Generation

To generate queries for multiple shapes at once, use `generate_all_shapes.py`. Edit the `SHAPES` dictionary and configuration at the top of the file, then run:

```bash
python generate_all_shapes.py
```

This produces one JSON file per shape, named `{DATASET}_{shape_name}.json`.

## Generation Methods

### In-Memory Generation
In-memory generation loads the entire RDF graph into memory and performs sampling directly on the in-memory structures. This approach:
* Is significantly faster, especially for larger queries
* Supports caching of search structures for faster repeated loading
* Requires sufficient memory to load the entire graph
* Is recommended for most use cases unless memory constraints prevent it
* DFS is mostly recommended, especially for large and/or dense graphs

For complex shapes, the generator builds both forward and reverse adjacency lists and uses DFS-based subgraph matching to find instances of the specified query topology in the graph.

### Endpoint-Based Generation
Endpoint-based generation queries a SPARQL endpoint to perform sampling. This approach:
* Has lower memory requirements
* Is slower, particularly for larger queries and often does not find large paths
* Requires a running SPARQL endpoint
* May be necessary for very large graphs that don't fit in memory

## Output
Generated queries are saved to files in the current directory with names following the pattern:

**Single mode:** `{DATASET}_{SHAPE}_{SIZE}_{timestamp}.txt`

**Batch mode:** `{DATASET}_SIZE_{SIZE}_{SHAPE}_{SIZE}_{timestamp}.txt`

When cardinality computation is enabled, results include both the query patterns and their cardinalities. 

## Citation

If you use this software in your research, please cite it as:

> Schwabe, T., & Acosta, M. (2026). *rdf-subgraph-sampler: Sampling Subgraph Templates
> and Cardinalities from RDF Knowledge Graphs* (Version 1.0.0) [Software].
> Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX

**BibTeX:**

```bibtex
@software{schwabe_rdf_subgraph_sampler_2026,
  author    = {Schwabe, Tim and Acosta, Maribel},
  title     = {{rdf-subgraph-sampler}: Sampling Subgraph Templates
               and Cardinalities from RDF Knowledge Graphs},
  year      = {2026},
  version   = {1.0.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://github.com/DE-TUM/rdf-subgraph-sampler}
}
```
