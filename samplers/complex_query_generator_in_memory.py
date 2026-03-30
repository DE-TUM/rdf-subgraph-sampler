#!/usr/bin/env python3
"""
Complex query generator that matches arbitrary query shape templates
against an RDF graph using DFS subgraph matching.

Shape string format:
    "?s B ?o, ?s2 B ?o, ?o B ?o2, ?o2 B ?z1, ?o2 ?/B ?z2/B"

Each comma-separated group is: <subject> <predicate_binding> <object>

Node tokens:
    ?name   -> always a variable
    name    -> always bound (instantiated with matched URI)
    ?name/B -> probabilistically bound (controlled by p_node)

Predicate tokens:
    B   -> always bound
    ?   -> always a variable
    ?/B -> probabilistically bound (controlled by p_edge)
"""

import os
import gzip
import json
import random
import re
import hashlib
from collections import defaultdict, Counter, deque
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set

import requests
from tqdm import tqdm

from samplers.dedup import dedup_key as _dedup_key

P_EDGE = 1.0
P_NODE = 0.3
SAVE_INTERVAL = 300
CARDINALITY_TIMEOUT = 3
CACHE_SUFFIX = ".complex_gen.cache.json"
MAX_CANDIDATES_PER_STEP = 50  # limit DFS branching on high-degree nodes

# ---------------------------------------------------------------------------
# Shape parsing
# ---------------------------------------------------------------------------

class TemplateTriple:
    __slots__ = ("subj_name", "subj_binding", "pred_binding", "obj_name", "obj_binding")

    def __init__(self, subj_name, subj_binding, pred_binding, obj_name, obj_binding):
        self.subj_name = subj_name
        self.subj_binding = subj_binding   # "var" | "bound" | "prob"
        self.pred_binding = pred_binding
        self.obj_name = obj_name
        self.obj_binding = obj_binding


def _parse_node_token(token):
    """Return (name, binding_type)."""
    token = token.strip()
    if token.startswith("?") and token.endswith("/B"):
        return token[1:-2], "prob"
    if token.startswith("?"):
        return token[1:], "var"
    if token.endswith("/B"):
        return token[:-2], "prob"
    return token, "bound"


def _parse_pred_token(token):
    token = token.strip()
    if token == "B":
        return "bound"
    if token == "?":
        return "var"
    if token == "?/B":
        return "prob"
    raise ValueError(f"Unknown predicate binding '{token}'. Use B, ?, or ?/B.")


def parse_shape(shape_string):
    """Parse shape string into list of TemplateTriple."""
    triples = []
    for part in shape_string.split(","):
        tokens = part.split()
        if len(tokens) != 3:
            raise ValueError(f"Expected 3 tokens in '{part.strip()}', got {len(tokens)}")
        subj_name, subj_binding = _parse_node_token(tokens[0])
        pred_binding = _parse_pred_token(tokens[1])
        obj_name, obj_binding = _parse_node_token(tokens[2])
        triples.append(TemplateTriple(subj_name, subj_binding, pred_binding, obj_name, obj_binding))
    return triples


def _get_template_nodes(triples):
    """Extract nodes with binding types; validate consistency across occurrences."""
    nodes = {}
    for t in triples:
        for name, binding in [(t.subj_name, t.subj_binding), (t.obj_name, t.obj_binding)]:
            if name in nodes:
                if nodes[name] != binding:
                    raise ValueError(f"Node '{name}' has inconsistent binding: "
                                     f"'{nodes[name]}' vs '{binding}'")
            else:
                nodes[name] = binding
    return nodes


def _compute_traversal_order(triples):
    """BFS over the template graph to produce an edge ordering where each edge
    has at least one already-visited endpoint.

    Returns (root_name, [edge_index, ...]).
    Root is picked as the highest-degree node that appears as a subject.
    """
    degrees = Counter()
    subject_nodes = set()
    for t in triples:
        degrees[t.subj_name] += 1
        degrees[t.obj_name] += 1
        subject_nodes.add(t.subj_name)

    candidates = [n for n in degrees if n in subject_nodes] or list(degrees.keys())
    root = max(candidates, key=lambda n: degrees[n])

    visited = {root}
    queue = deque([root])
    order = []
    remaining = set(range(len(triples)))

    while queue and remaining:
        node = queue.popleft()
        for idx in list(remaining):
            t = triples[idx]
            if t.subj_name == node:
                other = t.obj_name
            elif t.obj_name == node:
                other = t.subj_name
            else:
                continue
            remaining.remove(idx)
            order.append(idx)
            if other not in visited:
                visited.add(other)
                queue.append(other)

    if remaining:
        raise ValueError(f"Query shape is disconnected — edges {remaining} unreachable from root '{root}'.")

    return root, order


# ---------------------------------------------------------------------------
# Graph loading (forward + reverse adjacency)
# ---------------------------------------------------------------------------

_RE_NT = re.compile(r"^<([^>]+)>\s+<([^>]+)>\s+<([^>]+)>\s*\.\s*$")


def _load_graph(rdf_file, use_cache=True):
    """Return (adj, rev_adj) where adj[subj] = [(pred, obj)] and
    rev_adj[obj] = [(pred, subj)]."""
    cache_file = rdf_file + CACHE_SUFFIX
    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as fh:
                data = json.load(fh)
            adj = {k: [tuple(e) for e in v] for k, v in data["adj"].items()}
            rev = {k: [tuple(e) for e in v] for k, v in data["rev_adj"].items()}
            print(f"Loaded cached graph ({len(adj):,} subjects) from {cache_file}")
            return adj, rev
        except Exception as exc:
            print(f"Cache read failed ({exc}) — reparsing…")

    adj: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    rev: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    opener = gzip.open if rdf_file.endswith(".gz") else open
    total = 0
    with opener(rdf_file, "rt", encoding="utf-8", errors="ignore") as fh:
        for ln in fh:
            m = _RE_NT.match(ln)
            if m:
                s, p, o = m.group(1), m.group(2), m.group(3)
                adj[s].append((p, o))
                rev[o].append((p, s))
                total += 1
    print(f"Parsed {total:,} edges — {len(adj):,} subjects, {len(rev):,} objects.")

    if use_cache:
        try:
            with open(cache_file, "w") as fh:
                json.dump({"adj": dict(adj), "rev_adj": dict(rev)}, fh)
            print(f"Cached to {cache_file}")
        except Exception as exc:
            print(f"Could not write cache: {exc}")

    return dict(adj), dict(rev)


# ---------------------------------------------------------------------------
# Subgraph matching (DFS)
# ---------------------------------------------------------------------------

def _match_subgraph(adj, rev_adj, triples, traversal_order, root_name, start_node):
    """DFS match of template from *start_node* assigned to *root_name*.

    Returns (node_assignment, pred_assignment) or None.
    """
    stack = [(0, {root_name: start_node}, {}, {start_node})]

    while stack:
        trav_idx, node_assign, pred_assign, used = stack.pop()

        if trav_idx >= len(traversal_order):
            return node_assign, pred_assign  # complete match

        edge_idx = traversal_order[trav_idx]
        t = triples[edge_idx]
        s_known = t.subj_name in node_assign
        o_known = t.obj_name in node_assign

        if s_known and o_known:
            # verify edge exists
            s_uri, o_uri = node_assign[t.subj_name], node_assign[t.obj_name]
            preds = [p for p, o in adj.get(s_uri, []) if o == o_uri]
            if preds:
                p = random.choice(preds)
                stack.append((trav_idx + 1, node_assign, {**pred_assign, edge_idx: p}, used))

        elif s_known:
            # forward: find objects
            edges = list(adj.get(node_assign[t.subj_name], []))
            random.shuffle(edges)
            for p, o in edges[:MAX_CANDIDATES_PER_STEP]:
                if o not in used:
                    stack.append((trav_idx + 1,
                                  {**node_assign, t.obj_name: o},
                                  {**pred_assign, edge_idx: p},
                                  used | {o}))

        elif o_known:
            # backward: find subjects
            edges = list(rev_adj.get(node_assign[t.obj_name], []))
            random.shuffle(edges)
            for p, s in edges[:MAX_CANDIDATES_PER_STEP]:
                if s not in used:
                    stack.append((trav_idx + 1,
                                  {**node_assign, t.subj_name: s},
                                  {**pred_assign, edge_idx: p},
                                  used | {s}))
        # else: neither known → BFS ordering bug (should not happen)

    return None


# ---------------------------------------------------------------------------
# Query building
# ---------------------------------------------------------------------------

def _build_query(triples, node_assign, pred_assign, p_edge, p_node, graph_name=None):
    """Build SPARQL query from matched subgraph.

    Returns (query_str, [triple_pattern_str, ...], entities, bound_preds_set, bound_nodes_set).
    """
    patterns = []
    entities = []
    bound_preds: Set[int] = set()
    bound_nodes: Set[str] = set()

    for i, t in enumerate(triples):
        # subject
        s_uri = node_assign[t.subj_name]
        if t.subj_binding == "bound" or (t.subj_binding == "prob" and random.random() < p_node):
            s_str = f"<{s_uri}>"
            entities.append(s_uri)
            bound_nodes.add(t.subj_name)
        else:
            s_str = f"?{t.subj_name}"

        # predicate
        p_uri = pred_assign[i]
        if t.pred_binding == "bound" or (t.pred_binding == "prob" and random.random() < p_edge):
            p_str = f"<{p_uri}>"
            entities.append(p_uri)
            bound_preds.add(i)
        else:
            p_str = f"?p{i}"

        # object
        o_uri = node_assign[t.obj_name]
        if t.obj_binding == "bound" or (t.obj_binding == "prob" and random.random() < p_node):
            o_str = f"<{o_uri}>"
            entities.append(o_uri)
            bound_nodes.add(t.obj_name)
        else:
            o_str = f"?{t.obj_name}"

        patterns.append(f"{s_str} {p_str} {o_str}")

    wc = " . ".join(patterns)
    if graph_name:
        q = f"SELECT * FROM <{graph_name}> WHERE {{ {wc} }}"
    else:
        q = f"SELECT * WHERE {{ {wc} }}"

    return q, patterns, entities, bound_preds, bound_nodes


# ---------------------------------------------------------------------------
# Hashing / dedup
# ---------------------------------------------------------------------------

def _hash_query(triples, pred_assign, bound_preds, bound_nodes):
    """Structural hash: predicates used (at bound positions) + node binding pattern."""
    parts = []
    for i, t in enumerate(triples):
        pred = pred_assign[i] if i in bound_preds else "?"
        subj = "B" if t.subj_name in bound_nodes else "?"
        obj = "B" if t.obj_name in bound_nodes else "?"
        parts.append(f"{subj}|{pred}|{obj}")
    return hashlib.md5("||".join(parts).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Cardinality
# ---------------------------------------------------------------------------

def _get_cardinality(endpoint_url, where_clause, timeout=CARDINALITY_TIMEOUT):
    q = f"SELECT (COUNT(*) as ?count) WHERE {{ {where_clause} }}"
    try:
        r = requests.get(endpoint_url, params={"query": q, "format": "json"}, timeout=timeout)
        if r.status_code == 200:
            b = r.json().get("results", {}).get("bindings", [])
            if b and "count" in b[0]:
                return int(b[0]["count"]["value"])
    except Exception:
        pass
    return -1


# ---------------------------------------------------------------------------
# File saving
# ---------------------------------------------------------------------------

def _save_queries(generated, dataset_name, n_triples, ts, is_final=False, output_file=None):
    if output_file and is_final:
        fn = output_file
    else:
        tag = "_final" if is_final else ""
        fn = f"{dataset_name}_complex_{ts.strftime('%Y-%m-%d_%H-%M-%S')}_{n_triples}{tag}.json"
    with open(fn, "w") as fh:
        json.dump(generated, fh, indent=2)
    print(f"{'Final' if is_final else 'Partial'} save: {len(generated)} queries -> {fn}")
    return fn


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_queries(rdf_file,
                dataset_name,
                query_shape,
                n_queries=1000,
                endpoint_url=None,
                outfile=True,
                get_cardinality=False,
                use_cache=True,
                p_edge=P_EDGE,
                p_node=P_NODE,
                graph_name=None,
                output_file=None,
                dedup_method="hash"):
    """Generate *n_queries* unique queries matching *query_shape*.

    Args:
        rdf_file:       Path to .nt or .nt.gz file.
        dataset_name:   Tag for output filenames.
        query_shape:    Shape string, e.g. "?s B ?o, ?o B ?o2".
        n_queries:      Target number of unique queries.
        endpoint_url:   SPARQL endpoint (optional, for cardinality).
        outfile:        Save results to JSON files.
        get_cardinality: Compute COUNT(*) per query via endpoint.
        use_cache:      Cache parsed graph to disk.
        p_edge:         Probability of binding ?/B predicates.
        p_node:         Probability of binding ?/B nodes.
        graph_name:     Optional named graph URI.
        output_file:    Override output filename for final save.
    """
    if not rdf_file or not os.path.exists(rdf_file):
        raise FileNotFoundError(f"RDF file '{rdf_file}' not found")

    # --- parse template ---
    template = parse_shape(query_shape)
    nodes = _get_template_nodes(template)
    n_triples = len(template)
    root_name, traversal_order = _compute_traversal_order(template)

    print(f"Query shape: {query_shape}")
    print(f"  {n_triples} triple patterns, {len(nodes)} nodes")
    print(f"  Root: ?{root_name}, traversal order: {traversal_order}")
    print(f"  Node bindings: {nodes}")

    # --- load graph ---
    adj, rev_adj = _load_graph(rdf_file, use_cache)

    # --- prepare start candidates ---
    subjects = list(adj.keys())
    random.shuffle(subjects)

    print(f"Sampling {n_queries} unique complex queries …")
    now = datetime.now()
    generated: List[dict] = []
    seen_hashes: Set[str] = set()

    MAX_RESHUFFLES = 30
    MIN_NEW_PER_PASS = 5
    MAX_CONSECUTIVE_FAILURES = max(500, len(subjects) // 10)

    consecutive_failures = 0
    reshuffle_count = 0
    queries_at_start = 0
    last_save = 0
    subject_idx = 0

    with tqdm(total=n_queries, desc="Generating") as pbar:
        while len(generated) < n_queries:

            # --- end-of-pass logic ---
            if subject_idx >= len(subjects):
                new_in_pass = len(generated) - queries_at_start

                if new_in_pass == 0:
                    print(f"\nNo new queries in full pass. Dataset exhausted.")
                    print(f"Generated {len(generated)}/{n_queries} queries.")
                    break

                if new_in_pass < MIN_NEW_PER_PASS:
                    print(f"\nOnly {new_in_pass} new queries in last pass "
                          f"(threshold {MIN_NEW_PER_PASS}). Stopping.")
                    print(f"Generated {len(generated)}/{n_queries} queries.")
                    break

                if reshuffle_count >= MAX_RESHUFFLES:
                    print(f"\nReached max reshuffles ({MAX_RESHUFFLES}). Stopping.")
                    print(f"Generated {len(generated)}/{n_queries} queries.")
                    break

                random.shuffle(subjects)
                subject_idx = 0
                queries_at_start = len(generated)
                reshuffle_count += 1
                consecutive_failures = 0
                print(f"\nReshuffle #{reshuffle_count}: {new_in_pass} new queries in last pass, "
                      f"{len(generated)}/{n_queries} total …")

            # --- try matching from next start node ---
            start = subjects[subject_idx]
            subject_idx += 1

            result = _match_subgraph(adj, rev_adj, template, traversal_order,
                                     root_name, start)
            if result is None:
                continue

            node_assign, pred_assign = result

            # --- build query ---
            query_str, patterns, entities, bp, bn = \
                _build_query(template, node_assign, pred_assign, p_edge, p_node, graph_name)

            # --- dedup ---
            triple_list = [p.split() for p in patterns]
            h = _dedup_key(triple_list, dedup_method)
            if h in seen_hashes:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"\nWARNING: {consecutive_failures} consecutive duplicates. "
                          f"Generated {len(generated)}/{n_queries} queries.")
                    break
                continue

            consecutive_failures = 0
            seen_hashes.add(h)

            # --- cardinality ---
            y = -1
            if endpoint_url and get_cardinality:
                y = _get_cardinality(endpoint_url, " . ".join(patterns))

            generated.append({
                "query": query_str,
                "triples": triple_list,
                "x": entities,
                "y": y,
            })
            pbar.update(1)

            # --- periodic save ---
            if outfile and len(generated) - last_save >= SAVE_INTERVAL:
                _save_queries(generated, dataset_name, n_triples, now, output_file=output_file)
                last_save = len(generated)

    # --- final save ---
    if outfile and len(generated) > last_save:
        _save_queries(generated, dataset_name, n_triples, now, is_final=True, output_file=output_file)

    return generated


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Complex query generator with arbitrary shape templates")
    ap.add_argument("rdf_file", help="Path to .nt or .nt.gz file")
    ap.add_argument("dataset_name", help="Dataset tag for output filenames")
    ap.add_argument("--shape", required=True,
                    help='Query shape, e.g. "?s B ?o, ?o B ?o2"')
    ap.add_argument("--num", type=int, default=1000, help="Number of queries")
    ap.add_argument("--endpoint", help="SPARQL endpoint for cardinality")
    ap.add_argument("--card", action="store_true", help="Compute COUNT(*)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--p-edge", type=float, default=P_EDGE)
    ap.add_argument("--p-node", type=float, default=P_NODE)
    args = ap.parse_args()

    get_queries(
        rdf_file=args.rdf_file,
        dataset_name=args.dataset_name,
        query_shape=args.shape,
        n_queries=args.num,
        endpoint_url=args.endpoint,
        get_cardinality=args.card,
        use_cache=not args.no_cache,
        p_edge=args.p_edge,
        p_node=args.p_node,
    )
