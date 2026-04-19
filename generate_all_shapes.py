#!/usr/bin/env python3
"""Generate 1000 queries for each complex shape using the current RDF file and endpoint."""

from samplers import complex_query_generator_in_memory as gen

# --- Configuration ---
RDF_FILE = ""
ENDPOINT = "http://localhost:7001"
DATASET = ""
N_QUERIES = 5000
GET_CARDINALITY = True
USE_CACHE = True
P_EDGE = 1.0
P_NODE = 0.0

SHAPES = {
    "path_star_ends": "?s B ?o, ?s2 B ?o, ?o B ?o2, ?o2 B ?z1, ?o2 B ?z2",
    "flower":         "?a B ?l1, ?a B ?l2, ?a B ?b, ?b B ?m1, ?b B ?c, ?c B ?r1, ?c B ?r2",
    "binary_tree":    "?r B ?a, ?r B ?b, ?a B ?c, ?a B ?d, ?b B ?e, ?b B ?f",
    "cycle":          "?a B ?b, ?b B ?c, ?c B ?a",
    "diamond":        "?a B ?b, ?a B ?c, ?b B ?d, ?c B ?d",
    "snowflake":      "?center B ?a, ?center B ?b, ?center B ?c, ?center B ?d, ?a B ?l1, ?a B ?l2, ?b B ?r1, ?b B ?r2",
}

if __name__ == "__main__":
    for name, shape in SHAPES.items():
        print(f"\n{'='*60}")
        print(f"  Generating: {name}")
        print(f"  Shape: {shape}")
        print(f"{'='*60}\n")

        output_file = f"{DATASET}_{name}.json"

        try:
            results = gen.get_queries(
                rdf_file=RDF_FILE,
                dataset_name=DATASET,
                query_shape=shape,
                n_queries=N_QUERIES,
                endpoint_url=ENDPOINT if GET_CARDINALITY else None,
                outfile=True,
                get_cardinality=GET_CARDINALITY,
                use_cache=USE_CACHE,
                p_edge=P_EDGE,
                p_node=P_NODE,
                output_file=output_file,
            )
            print(f"\n  => {name}: {len(results)} queries -> {output_file}")
        except Exception as e:
            print(f"\n  => {name}: FAILED — {e}")

    print(f"\n{'='*60}")
    print("  All shapes complete.")
    print(f"{'='*60}")
