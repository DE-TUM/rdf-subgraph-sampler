"""Remove queries with failed cardinality (y == -1) from complex query JSON files.

Usage:
    python clean_queries.py /path/to/data
    python clean_queries.py  # defaults to ../fully-inductive-cardinality-estimation/data
"""

import json
import sys
from pathlib import Path

COMPLEX_SHAPES = ["cycle", "diamond", "flower", "snowflake", "binary_tree", "path_star_ends"]

def clean_queries(data_root: Path):
    query_dirs = sorted(data_root.glob("*_queries"))

    for qdir in query_dirs:
        # Only process complex query folders
        if not any(f"_{shape}_queries" in qdir.name for shape in COMPLEX_SHAPES):
            continue

        raw_file = qdir / "raw" / "queries.json"
        if not raw_file.exists():
            print(f"SKIP {qdir.name}: no raw/queries.json")
            continue

        with open(raw_file, "r") as f:
            queries = json.load(f)

        before = len(queries)
        cleaned = [q for q in queries if q.get("y") != -1]
        removed = before - len(cleaned)

        if removed > 0:
            with open(raw_file, "w") as f:
                json.dump(cleaned, f)
            print(f"{qdir.name}: {before} -> {len(cleaned)} (removed {removed})")
        else:
            print(f"{qdir.name}: {before} queries, all clean")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
    else:
        root = Path(__file__).parent / ".." / "fully-inductive-cardinality-estimation" / "data"

    root = root.resolve()
    print(f"Scanning: {root}\n")
    clean_queries(root)
