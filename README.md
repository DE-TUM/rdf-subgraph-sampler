# rdf-subgraph-sampler 
Sampling approach to obtain conjunctive queries from RDF knowledge graphs and their cardinality. The tool supports both endpoint-based and in-memory generation methods for extracting star and path-shaped query patterns.

## Preparation 
* Clone the repository
* Install the required Python libraries: `tqdm` and `requests`
* For endpoint-based generation: Load the RDF knowledge graph in a SPARQL endpoint
* For in-memory generation: Have your RDF file available locally (supports .nt format)

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
SHAPE = "path"  # Shape of subgraphs to generate: "path", "star", "flower", "snowflake"
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
```

### `batch_sampler_config.py`
Configuration file for batch query generation (`batch_sampler.py`). Includes all standard parameters plus batch-specific configurations.

**Example:**
```python
# Configuration file for batch_sampler.py

# General configuration
ENDPOINT = "http://localhost:8890/sparql"  # URL of the SPARQL endpoint where the KG is hosted (required)
SHAPE = "path"  # Shape of subgraphs to generate: "path", "star", "flower", "snowflake"
DATASET = "LUBM_TEST"  # Dataset name (used for output file)
USE_CACHE = True  # Whether to use cached search structures for in memory generation
IN_MEMORY = True  # Whether to use in-memory search and generation instead of sparql endpoint

RDF_FILE_PATH = "....nt"  # Path to RDF file to use

GET_CARDINALITY = True  # Whether to compute cardinality for generated queries in-memory

# Star-specific configuration
# NOTE: MIN_OBJECTS_INSTANTIATED and MAX_OBJECTS_INSTANTIATED are calculated dynamically
# based on query size (min=0, max=ceil(0.5 * size)) in batch sampling
P_PREDICATE = 1.0  # Probability of instantiating predicates (1.0 = always instantiate, 0.0 = never instantiate)

# Path-specific configuration
SAMPLING_METHOD = "dfs"  # Sampling method: "dfs", "bfs" or "random_walk"
P_EDGE = 1.0  # Probability of instantiating a predicate in path queries (1.0 = always instantiate, 0.0 = never instantiate)
P_NODE = 0.3  # Probability of instantiating an intermediate node in path queries
P_START_END = 0.3  # Probability of instantiating start/end node (non-seed) in path queries

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
* `SHAPE`: Shape of subgraphs to generate - `"path"` or `"star"`
* `DATASET`: Dataset name used for output file naming
* `USE_CACHE`: Whether to use cached search structures for faster generation (default: True) - only for in-memory Generation
* `IN_MEMORY`: Whether to use in-memory generation instead of SPARQL endpoint queries (default: True)
* `RDF_FILE_PATH`: Path to the RDF file for in-memory generation (required when `IN_MEMORY=True`)
* `GET_CARDINALITY`: Whether to compute cardinality for generated queries if in-memory(default: True)

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

### Batch-Specific Parameters
* `QUERY_CONFIGURATIONS`: List of `(size, number_of_queries)` tuples defining the batch configurations

## Batch Sampler

The batch sampler (`batch_sampler.py`) allows you to generate multiple sets of queries with different sizes in a single run. This is particularly useful for:

* **Experimental workflows**: Generate comprehensive datasets with various query complexities
* **Performance testing**: Create queries of different sizes for benchmarking
* **Systematic evaluation**: Ensure consistent generation across multiple configurations

### Key Features:
* **Automatic size-based naming**: Output files include size information (e.g., `LUBM_TEST_SIZE_3_timestamp.txt`)
* **Dynamic object instantiation**: For star queries, min/max objects are automatically calculated based on query size
* **Error resilience**: If one configuration fails, the process continues with remaining configurations
* **Progress tracking**: Clear output showing completion status for each configuration

### Example Batch Configuration:
```python
QUERY_CONFIGURATIONS = [
    (3, 50),    # 50 queries of size 3 (max objects: 2)
    (5, 100),   # 100 queries of size 5 (max objects: 3) 
    (8, 75),    # 75 queries of size 8 (max objects: 4)
    (12, 25),   # 25 queries of size 12 (max objects: 6)
]
```

This configuration will generate 4 separate output files, each containing queries of the specified size and count.

## Generation Methods

### In-Memory Generation
In-memory generation loads the entire RDF graph into memory and performs sampling directly on the in-memory structures. This approach:
* Is significantly faster, especially for larger queries
* Supports caching of search structures for faster repeated loading
* Requires sufficient memory to load the entire graph
* Is recommended for most use cases unless memory constraints prevent it
* DFS is mostly recommended, especially for large and/or dense graphs

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

