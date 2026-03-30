"""
Deduplication utilities for BGP (Basic Graph Pattern) queries.

Provides two methods:
  - "hash": fast multiset-based hash (ignores join structure, correct for stars)
  - "wl":   WL-1 canonical form (respects join structure, correct for any shape)
"""

import hashlib
from collections import defaultdict


def is_variable(term):
    return isinstance(term, str) and term.startswith("?")


# ── Hash-based dedup (multiset of predicate/object patterns) ────────

def _hash_dedup_key(triples):
    """MD5 of sorted (pred, bound_obj_or_?) parts.  Fast but ignores joins."""
    parts = []
    for triple in triples:
        _subj, pred, obj = triple
        if is_variable(obj):
            parts.append(f"{pred}=?")
        else:
            parts.append(f"{pred}={obj}")
    parts.sort()
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()


# ── WL-based dedup (true BGP isomorphism) ───────────────────────────

def canonicalize_bgp(triples):
    """
    Canonical form of a BGP under variable renaming.

    Uses WL-1 color refinement followed by individualize-and-refine.
    Returns a hashable tuple-of-tuples: two BGPs are isomorphic iff
    their canonical forms are equal.
    """
    triples = [tuple(t) for t in triples]

    all_vars = list({t for tri in triples for t in tri if is_variable(t)})

    if not all_vars:
        return tuple(sorted(triples))

    def apply_renaming(mapping):
        out = [tuple(mapping.get(t, t) for t in tri) for tri in triples]
        return tuple(sorted(out))

    def normalize(colors):
        unique = sorted(set(colors.values()))
        cmap = {c: i for i, c in enumerate(unique)}
        return {v: cmap[c] for v, c in colors.items()}

    def initial_colors():
        colors = {}
        for var in all_vars:
            occs = []
            for tri in triples:
                for pos, term in enumerate(tri):
                    if term == var:
                        cctx = tuple(sorted(
                            (p, t) for p, t in enumerate(tri)
                            if p != pos and not is_variable(t)
                        ))
                        vctx = tuple(sorted(
                            p for p, t in enumerate(tri)
                            if p != pos and is_variable(t)
                        ))
                        occs.append((pos, cctx, vctx))
            colors[var] = tuple(sorted(occs))
        return colors

    def refine(colors):
        colors = normalize(colors)
        for _ in range(len(all_vars)):
            new = {}
            for var in all_vars:
                nbr = []
                for tri in triples:
                    for pos, term in enumerate(tri):
                        if term == var:
                            sig = tuple(sorted(
                                (p, 'V', colors[ot]) if is_variable(ot)
                                else (p, 'C', ot)
                                for p, ot in enumerate(tri) if p != pos
                            ))
                            nbr.append((pos, sig))
                new[var] = (colors[var], tuple(sorted(nbr)))
            new = normalize(new)
            if new == colors:
                break
            colors = new
        return colors

    def search(colors):
        groups = defaultdict(list)
        for v in all_vars:
            groups[colors[v]].append(v)
        ordered = [sorted(groups[c]) for c in sorted(groups)]

        if all(len(g) == 1 for g in ordered):
            # Assign canonical names by first-appearance in sorted abstract
            # triples (variables replaced by color placeholders).  This is
            # consistent (same for isomorphic inputs) but not guaranteed to
            # be lex-minimal in all cases — that's fine for deduplication.
            abstract = []
            for tri in triples:
                a_tri = tuple(f"?_c{colors[t]}" if is_variable(t) else t for t in tri)
                abstract.append(a_tri)
            sorted_indices = sorted(range(len(triples)), key=lambda i: abstract[i])
            mapping = {}
            counter = 0
            for idx in sorted_indices:
                for term in triples[idx]:
                    if is_variable(term) and term not in mapping:
                        mapping[term] = f"?v{counter}"
                        counter += 1
            return apply_renaming(mapping)

        target = min((g for g in ordered if len(g) > 1), key=len)
        best = None
        for var in target:
            nc = dict(colors)
            nc[var] = max(colors.values()) + 1
            nc = refine(nc)
            result = search(nc)
            if best is None or result < best:
                best = result
        return best

    return search(refine(initial_colors()))


def _wl_dedup_key(triples):
    """MD5 of WL canonical form.  Precise but slightly slower."""
    canon = canonicalize_bgp(triples)
    return hashlib.md5(str(canon).encode("utf-8")).hexdigest()


# ── Public API ──────────────────────────────────────────────────────

def dedup_key(triples, method="hash"):
    """
    Compute a deduplication key for a list of [subject, predicate, object] patterns.

    Args:
        triples: list of [s, p, o] lists/tuples
        method: "hash" (fast, multiset-based) or "wl" (precise, isomorphism-based)

    Returns:
        A hashable string key.  Equal keys ⟹ duplicate query.
    """
    if method == "wl":
        return _wl_dedup_key(triples)
    return _hash_dedup_key(triples)
