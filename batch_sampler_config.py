# Configuration file for batch_sampler.py

# General configuration
ENDPOINT = "http://localhost:7001"  # URL of the SPARQL endpoint where the KG is hosted (required)
SHAPE = "complex"  # Shape of subgraphs to generate: "path", "star", "complex", "flower", "snowflake"
DATASET = "wn18rr_v4"  # Dataset name (used for output file)
USE_CACHE = True  # Whether to use cached search structures for in memory generation
IN_MEMORY = True  # Whether to use in-memory search and generation instead of sparql endpoint
GRAPH_NAME = None # The named graph name used for querying the endpoint

RDF_FILE_PATH = "/home/tim/fully-inductive-cardinality-estimation/data/wn18rr_v4/raw/59622641"  # Path to RDF file to use

GET_CARDINALITY = True  # Whether to compute cardinality for generated queries in-memory

# Star-specific configuration
# NOTE: MIN_OBJECTS_INSTANTIATED and MAX_OBJECTS_INSTANTIATED are calculated dynamically
# based on query size (min=0, max=ceil(0.5 * size)) and no longer used from here
# Complex-specific configuration
# Shape string for complex queries, e.g. "?s B ?o, ?s2 B ?o, ?o B ?o2, ?o2 B ?z1, ?o2 ?/B ?z2/B"
# Node tokens: ?name (variable), name (bound), ?name/B (probabilistic)
# Predicate tokens: B (bound), ? (variable), ?/B (probabilistic)
# --- Example shapes ---
#
# Path with star on each end (5 triples):
#   ?s ---> ?o <--- ?s2       ?o2 ---> ?z1
#            |                  |
#            v                  v
#           ?o2               ?z2
#QUERY_SHAPE = "?s B ?o, ?s2 B ?o, ?o B ?o2, ?o2 B ?z1, ?o2 B ?z2"
#
# Simple path (3 triples):
#   ?a ---> ?b ---> ?c ---> ?d
# QUERY_SHAPE = "?a B ?b, ?b B ?c, ?c B ?d"
#
# Star (4 triples, center = ?c):
#   ?a <--- ?c ---> ?b
#           |
#           v
#          ?d ---> ?e
# QUERY_SHAPE = "?c B ?a, ?c B ?b, ?c B ?d, ?d B ?e"
#
# Flower — path with stars at each node (7 triples):
#   ?l1 <--- ?a ---> ?b ---> ?c ---> ?r1
#            ^       |               |
#           ?l2     ?m1             ?r2
# QUERY_SHAPE = "?a B ?l1, ?a B ?l2, ?a B ?b, ?b B ?m1, ?b B ?c, ?c B ?r1, ?c B ?r2"
#
# Binary tree depth 3 (6 triples, root = ?r):
#          ?r
#         / \
#       ?a   ?b
#       /\   /\
#     ?c ?d ?e ?f
#QUERY_SHAPE = "?r B ?a, ?r B ?b, ?a B ?c, ?a B ?d, ?b B ?e, ?b B ?f"
#
# Triangle / cycle (3 triples):
#   ?a ---> ?b ---> ?c ---> ?a
#QUERY_SHAPE = "?a B ?b, ?b B ?c, ?c B ?a"
#
# Diamond (4 triples):
#     ?a ---> ?b
#     |        |
#     v        v
#     ?c ---> ?d
#QUERY_SHAPE = "?a B ?b, ?a B ?c, ?b B ?d, ?c B ?d"
#
# Snowflake — star with branches (8 triples):
#   ?l1 <--- ?a          ?b ---> ?r1
#   ?l2 <--- ?a          ?b ---> ?r2
#             \         /
#              ?center
#             /         \
#   ?l3 <--- ?c          ?d ---> ?r3
#   ?l4 <--- ?c          ?d ---> ?r4
QUERY_SHAPE = "?center B ?a, ?center B ?b, ?center B ?c, ?center B ?d, ?a B ?l1, ?a B ?l2, ?b B ?r1, ?b B ?r2"

P_PREDICATE = 1.0  # Probability of instantiating predicates (1.0 = always instantiate, 0.0 = never instantiate)

# Path-specific configuration
SAMPLING_METHOD = "dfs"  # Sampling method: "dfs", "bfs" or "random_walk"
P_EDGE = 1.0  # Probability of instantiating a predicate in path queries (1.0 = always instantiate, 0.0 = never instantiate)
P_NODE = 0.0  # Probability of instantiating an intermediate node in path queries
P_START_END = 0.3  # Probability of instantiating start/end node (non-seed) in path queries

# ====== BATCH CONFIGURATION ======
# Specify multiple query sizes and how many queries to generate for each size
# Format: [(size, number_of_queries), (size, number_of_queries), ...]
QUERY_CONFIGURATIONS = [
    (5, 100),    # size is ignored for complex shape (derived from QUERY_SHAPE), just set n_queries
] 

