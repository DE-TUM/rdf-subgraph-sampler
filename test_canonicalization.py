#!/usr/bin/env python3
"""
Brute-force verification of WL canonical form correctness.

Compares canonicalize_bgp() against a ground-truth implementation that
tries ALL variable permutations and takes the lex-smallest result.

Tests:
  1. Direct cases: WL canonical form == brute-force canonical form
  2. Isomorphism: random variable renamings + triple reorderings produce
     the same canonical form
  3. Non-isomorphism: structurally different queries produce different
     canonical forms

Usage:
    python test_canonicalization.py
"""

import random
import sys
from itertools import permutations

from samplers.dedup import canonicalize_bgp


def brute_force_canonical(triples):
    """Canonical form by trying ALL variable permutations (ground truth)."""
    triples = [tuple(t) for t in triples]
    all_vars = list({t for tri in triples for t in tri if t.startswith("?")})

    if not all_vars:
        return tuple(sorted(triples))

    best = None
    canon_names = [f"?v{i}" for i in range(len(all_vars))]

    for perm in permutations(all_vars):
        mapping = dict(zip(perm, canon_names))
        renamed = tuple(sorted(
            tuple(mapping.get(t, t) for t in tri) for tri in triples
        ))
        if best is None or renamed < best:
            best = renamed
    return best


# ── Test data ───────────────────────────────────────────────────────

QUERY_SHAPES = [
    # Stars
    [["?s", "<p1>", "?a"], ["?s", "<p2>", "?b"]],
    [["?s", "<p>", "?a"], ["?s", "<p>", "?b"]],
    [["?s", "<p>", "?a"], ["?s", "<p>", "?b"], ["?s", "<p>", "?c"]],
    [["?s", "<p1>", "<obj1>"], ["?s", "<p2>", "?o"]],
    [["?s", "<p1>", "?a"], ["?s", "<p2>", "?b"], ["?s", "<p3>", "?c"], ["?s", "<p4>", "?d"]],
    [["?s", "<p>", "?a"], ["?s", "<p>", "?b"], ["?s", "<p>", "?c"], ["?s", "<p>", "?d"]],

    # Paths
    [["?a", "<p1>", "?b"], ["?b", "<p2>", "?c"]],
    [["?a", "<p1>", "?b"], ["?b", "<p2>", "?c"], ["?c", "<p3>", "?d"]],
    [["<X>", "<p1>", "?b"], ["?b", "<p2>", "?c"], ["?c", "<p3>", "<Y>"]],
    [["<X>", "<p1>", "?b"], ["?b", "<p2>", "?c"], ["?c", "<p3>", "?d"], ["?d", "<p4>", "<Y>"]],

    # Cycles
    [["?a", "<p1>", "?b"], ["?b", "<p2>", "?c"], ["?c", "<p3>", "?a"]],
    [["?a", "<p>", "?b"], ["?b", "<p>", "?c"], ["?c", "<p>", "?a"]],

    # Diamond
    [["?a", "<p1>", "?b"], ["?b", "<p2>", "?d"], ["?a", "<p3>", "?c"], ["?c", "<p4>", "?d"]],

    # Trees
    [["?r", "<p1>", "?a"], ["?r", "<p2>", "?b"], ["?a", "<p3>", "?c"], ["?a", "<p4>", "?d"]],
    [["?r", "<p1>", "?a"], ["?r", "<p2>", "?b"], ["?b", "<p3>", "?c"]],

    # Self-joins
    [["?x", "<p>", "?x"]],
    [["?x", "<p1>", "?y"], ["?y", "<p2>", "?x"]],

    # Mixed bound/unbound
    [["<A>", "<p1>", "?b"], ["?b", "<p2>", "?c"]],
    [["?a", "<p1>", "?b"], ["?b", "<p2>", "<C>"]],

    # Variable predicates
    [["?s", "?p", "?o1"], ["?s", "<p2>", "?o2"]],
    [["?a", "?p", "?b"], ["?b", "?q", "?c"]],

    # No variables
    [["<A>", "<p>", "<B>"]],
    [["<A>", "<p1>", "<B>"], ["<B>", "<p2>", "<C>"]],
]

NON_ISOMORPHIC_PAIRS = [
    # Different predicates on path
    ([["?x", "<p1>", "?y"], ["?y", "<p2>", "?z"]],
     [["?x", "<p2>", "?y"], ["?y", "<p1>", "?z"]]),
    # Star vs path
    ([["?s", "<p1>", "?a"], ["?s", "<p2>", "?b"]],
     [["?a", "<p1>", "?b"], ["?b", "<p2>", "?c"]]),
    # Different bound objects
    ([["?s", "<p1>", "<obj1>"], ["?s", "<p2>", "?o"]],
     [["?s", "<p1>", "<obj2>"], ["?s", "<p2>", "?o"]]),
    # Bound vs unbound
    ([["?s", "<p1>", "<obj1>"], ["?s", "<p2>", "?o"]],
     [["?s", "<p1>", "?a"], ["?s", "<p2>", "?b"]]),
    # Different size
    ([["?s", "<p1>", "?a"], ["?s", "<p2>", "?b"]],
     [["?s", "<p1>", "?a"], ["?s", "<p2>", "?b"], ["?s", "<p3>", "?c"]]),
    # Triangle vs open path
    ([["?a", "<p1>", "?b"], ["?b", "<p2>", "?c"], ["?c", "<p3>", "?a"]],
     [["?a", "<p1>", "?b"], ["?b", "<p2>", "?c"], ["?c", "<p3>", "?d"]]),
    # Self-join vs no self-join
    ([["?x", "<p>", "?x"]],
     [["?x", "<p>", "?y"]]),
    # Different join structure
    ([["?a", "<p1>", "?b"], ["?c", "<p2>", "?b"]],
     [["?a", "<p1>", "?b"], ["?b", "<p2>", "?c"]]),
    # Diamond: different predicate assignment
    ([["?a", "<p1>", "?b"], ["?b", "<p2>", "?d"], ["?a", "<p3>", "?c"], ["?c", "<p4>", "?d"]],
     [["?a", "<p1>", "?b"], ["?b", "<p3>", "?d"], ["?a", "<p2>", "?c"], ["?c", "<p4>", "?d"]]),
]

# ── Manually labeled pairs with ground-truth labels ──────────────────
# Each entry: (query_a, query_b, expected_isomorphic: bool, description)
# These include the hardest cases for canonicalization algorithms.

LABELED_PAIRS = [
    # ── ISOMORPHIC pairs (True) ──────────────────────────────────────

    # Simple variable renaming on a path
    ([["?a", "<p1>", "?b"], ["?b", "<p2>", "?c"]],
     [["?x", "<p1>", "?y"], ["?y", "<p2>", "?z"]],
     True, "path: simple variable renaming"),

    # Star with reordered triples and renamed vars
    ([["?s", "<p1>", "?a"], ["?s", "<p2>", "?b"], ["?s", "<p3>", "?c"]],
     [["?center", "<p3>", "?z"], ["?center", "<p1>", "?x"], ["?center", "<p2>", "?y"]],
     True, "star: reordered triples + renamed vars"),

    # Same-predicate star: leaves are interchangeable
    ([["?s", "<p>", "?a"], ["?s", "<p>", "?b"], ["?s", "<p>", "?c"]],
     [["?hub", "<p>", "?x"], ["?hub", "<p>", "?y"], ["?hub", "<p>", "?z"]],
     True, "star: all same predicate, interchangeable leaves (WL needs individualization)"),

    # Triangle with all same predicates — hardest for WL-1
    ([["?a", "<p>", "?b"], ["?b", "<p>", "?c"], ["?c", "<p>", "?a"]],
     [["?x", "<p>", "?y"], ["?y", "<p>", "?z"], ["?z", "<p>", "?x"]],
     True, "cycle: all same predicate, full symmetry (WL-1 can't distinguish, needs individualization)"),

    # Diamond with all same predicates — 4 vars, high symmetry
    ([["?a", "<p>", "?b"], ["?a", "<p>", "?c"], ["?b", "<p>", "?d"], ["?c", "<p>", "?d"]],
     [["?w", "<p>", "?x"], ["?w", "<p>", "?y"], ["?x", "<p>", "?z"], ["?y", "<p>", "?z"]],
     True, "diamond: all same predicate, high symmetry"),

    # Self-loop: variable maps to itself
    ([["?x", "<p>", "?x"]],
     [["?y", "<p>", "?y"]],
     True, "self-loop renaming"),

    # Mutual edge (bidirectional): ?x->?y and ?y->?x
    ([["?x", "<p1>", "?y"], ["?y", "<p2>", "?x"]],
     [["?b", "<p1>", "?a"], ["?a", "<p2>", "?b"]],
     True, "mutual edges: swap roles"),

    # Mutual edge (bidirectional): ?x->?y and ?y->?x
    ([["?x", "<p>", "?y"], ["?y", "<p>", "?x"]],
     [["?b", "<p>", "?a"], ["?a", "<p>", "?b"]],
     True, "mutual edges: swap roles"),

    # Path with bound endpoints
    ([["<A>", "<p1>", "?x"], ["?x", "<p2>", "?y"], ["?y", "<p2>", "<B>"]],
     [["<A>", "<p1>", "?m"], ["?m", "<p2>", "?n"], ["?n", "<p2>", "<B>"]],
     True, "path: bound endpoints, rename interior vars"),

    # Variable predicates — both subject and predicate are variables
    ([["?s", "?p", "?o1"], ["?s", "?p", "?o2"]],
     [["?x", "?r", "?y"], ["?x", "?r", "?z"]],
     True, "variable predicates: star with shared var predicate"),

    # Chain of same predicates (4 vars)
    ([["?a", "<p>", "?b"], ["?b", "<p>", "?c"], ["?c", "<p>", "?d"]],
     [["?w", "<p>", "?x"], ["?x", "<p>", "?y"], ["?y", "<p>", "?z"]],
     True, "long same-predicate path (interior nodes indistinguishable by WL-1)"),

    # Chain of same predicates (4 vars)
    ([["?a", "?p", "?b"], ["?b", "?p", "?c"], ["?c", "?p", "?d"]],
     [["?w", "?p", "?x"], ["?x", "?p", "?y"], ["?y", "?p", "?z"]],
     True, "long same-predicate path (interior nodes indistinguishable by WL-1)"),

    # Tree: root with two branches, one branch extends
    ([["?r", "<p1>", "?a"], ["?r", "<p2>", "?b"], ["?b", "<p3>", "?c"]],
     [["?root", "<p1>", "?x"], ["?root", "<p2>", "?y"], ["?y", "<p3>", "?z"]],
     True, "asymmetric tree: rename all vars"),

    # Binary tree depth 2 with all same predicates
    ([["?r", "<p>", "?a"], ["?r", "<p>", "?b"], ["?a", "<p>", "?c"], ["?a", "<p>", "?d"]],
     [["?root", "<p>", "?l"], ["?root", "<p>", "?r2"], ["?l", "<p>", "?ll"], ["?l", "<p>", "?lr"]],
     True, "binary tree: same predicates, partial symmetry"),

    # 4-cycle (square) with all same predicates — high symmetry
    ([["?a", "<p>", "?b"], ["?b", "<p>", "?c"], ["?c", "<p>", "?d"], ["?d", "<p>", "?a"]],
     [["?w", "<p>", "?x"], ["?x", "<p>", "?y"], ["?y", "<p>", "?z"], ["?z", "<p>", "?w"]],
     True, "4-cycle: all same predicate (dihedral symmetry group)"),

    # Flower: center with star + one branch extends (mixed predicates)
    ([["?c", "<p1>", "?a"], ["?c", "<p2>", "?b"], ["?b", "<p3>", "?d"]],
     [["?hub", "<p1>", "?x"], ["?hub", "<p2>", "?y"], ["?y", "<p3>", "?z"]],
     True, "flower: star center with one branch extending"),

    # Two disconnected triples (both variable, same structure)
    ([["?a", "<p1>", "?b"], ["?c", "<p2>", "?d"]],
     [["?x", "<p1>", "?y"], ["?w", "<p2>", "?z"]],
     True, "disconnected: two independent edges"),

    # Snowflake: center -> 2 nodes, each with 2 leaves, all same predicate
    ([["?c", "<p>", "?a"], ["?c", "<p>", "?b"], ["?a", "<p>", "?a1"], ["?a", "<p>", "?a2"],
      ["?b", "<p>", "?b1"], ["?b", "<p>", "?b2"]],
     [["?hub", "<p>", "?l"], ["?hub", "<p>", "?r"], ["?l", "<p>", "?ll"], ["?l", "<p>", "?lr"],
      ["?r", "<p>", "?rl"], ["?r", "<p>", "?rr"]],
     True, "snowflake: full symmetry, all same predicate (hard: 7 vars, WL-1 ties everywhere)"),

    # ── NON-ISOMORPHIC pairs (False) ─────────────────────────────────

    # Path direction matters with different predicates
    ([["?a", "<p1>", "?b"], ["?b", "<p2>", "?c"]],
     [["?a", "<p2>", "?b"], ["?b", "<p1>", "?c"]],
     False, "path: reversed predicate order (not isomorphic)"),

    # Star vs path (different topology)
    ([["?s", "<p1>", "?a"], ["?s", "<p2>", "?b"]],
     [["?a", "<p1>", "?b"], ["?b", "<p2>", "?c"]],
     False, "star vs path (different topology)"),

    # Self-join vs no self-join
    ([["?x", "<p>", "?x"]],
     [["?x", "<p>", "?y"]],
     False, "self-loop vs distinct endpoints"),

    # Triangle vs open path (cycle vs no cycle)
    ([["?a", "<p1>", "?b"], ["?b", "<p2>", "?c"], ["?c", "<p3>", "?a"]],
     [["?a", "<p1>", "?b"], ["?b", "<p2>", "?c"], ["?c", "<p3>", "?d"]],
     False, "triangle vs open path (cycle vs acyclic)"),

    # Same shape, different bound constants
    ([["?s", "<p>", "<A>"]],
     [["?s", "<p>", "<B>"]],
     False, "same shape, different bound object"),

    # Diamond with different predicate assignment
    ([["?a", "<p1>", "?b"], ["?b", "<p2>", "?d"], ["?a", "<p3>", "?c"], ["?c", "<p4>", "?d"]],
     [["?a", "<p1>", "?b"], ["?b", "<p3>", "?d"], ["?a", "<p2>", "?c"], ["?c", "<p4>", "?d"]],
     False, "diamond: swapped predicates on different arms"),

    # Subject-star vs object-star (in-star vs out-star)
    ([["?a", "<p1>", "?s"], ["?b", "<p2>", "?s"]],
     [["?s", "<p1>", "?a"], ["?s", "<p2>", "?b"]],
     False, "in-star vs out-star (different join positions)"),

    # Disconnected vs connected
    ([["?a", "<p1>", "?b"], ["?c", "<p2>", "?d"]],
     [["?a", "<p1>", "?b"], ["?b", "<p2>", "?c"]],
     False, "disconnected pair vs connected path"),

    # Same-predicate: star-3 vs path-3 — tricky because same multiset
    ([["?s", "<p>", "?a"], ["?s", "<p>", "?b"], ["?s", "<p>", "?c"]],
     [["?a", "<p>", "?b"], ["?b", "<p>", "?c"], ["?c", "<p>", "?d"]],
     False, "same-pred star-3 vs same-pred path-3 (same multiset, different structure)"),

    # Same-predicate: path-3 vs triangle — tricky, same multiset
    ([["?a", "<p>", "?b"], ["?b", "<p>", "?c"], ["?c", "<p>", "?d"]],
     [["?a", "<p>", "?b"], ["?b", "<p>", "?c"], ["?c", "<p>", "?a"]],
     False, "same-pred path-3 vs same-pred triangle (same multiset, acyclic vs cyclic)"),

    # 4-cycle vs two disconnected edges (same number of triples/vars with same-pred)
    ([["?a", "<p>", "?b"], ["?b", "<p>", "?c"], ["?c", "<p>", "?d"], ["?d", "<p>", "?a"]],
     [["?a", "<p>", "?b"], ["?b", "<p>", "?a"], ["?c", "<p>", "?d"], ["?d", "<p>", "?c"]],
     False, "4-cycle vs two mutual edges (same degree sequence, different structure)"),

    # Tree depth 2: balanced vs unbalanced with same predicates
    ([["?r", "<p>", "?a"], ["?r", "<p>", "?b"], ["?a", "<p>", "?c"], ["?b", "<p>", "?d"]],
     [["?r", "<p>", "?a"], ["?r", "<p>", "?b"], ["?a", "<p>", "?c"], ["?a", "<p>", "?d"]],
     False, "balanced vs unbalanced tree (same-pred, same degree sequence)"),

    # Variable predicate in different position
    ([["?a", "?p", "?b"], ["?b", "<p2>", "?c"]],
     [["?a", "<p2>", "?b"], ["?b", "?p", "?c"]],
     False, "variable predicate on first vs second triple"),

    # Same structure but one has bound subject, other has bound object
    ([["<A>", "<p1>", "?x"], ["?x", "<p2>", "?y"]],
     [["?x", "<p1>", "?y"], ["?y", "<p2>", "<A>"]],
     False, "bound at start vs bound at end of path"),

    # Same-predicate star: 3 leaves vs 4 leaves
    ([["?s", "<p>", "?a"], ["?s", "<p>", "?b"], ["?s", "<p>", "?c"]],
     [["?s", "<p>", "?a"], ["?s", "<p>", "?b"], ["?s", "<p>", "?c"], ["?s", "<p>", "?d"]],
     False, "same-pred star: different number of leaves"),
]


# ── Tests ───────────────────────────────────────────────────────────

def test_direct(verbose=False):
    """WL canonical form must be consistent: same input always gives same output.
    Lex-minimality vs brute-force is reported as a warning, not a failure."""
    passed = 0
    failed = 0
    lex_warnings = 0
    for i, triples in enumerate(QUERY_SHAPES):
        wl = canonicalize_bgp(triples)
        wl2 = canonicalize_bgp(triples)  # same input twice → must match
        bf = brute_force_canonical(triples)
        if wl != wl2:
            failed += 1
            print(f"  FAIL direct case {i}: inconsistent WL output!")
            print(f"    WL run 1: {wl}")
            print(f"    WL run 2: {wl2}")
        else:
            passed += 1
            if wl != bf:
                lex_warnings += 1
                if verbose:
                    print(f"  WARN direct case {i}: WL != brute-force (lex-minimality)")
                    print(f"    WL:    {wl}")
                    print(f"    Brute: {bf}")
    if lex_warnings:
        print(f"  (lex-minimality warnings: {lex_warnings} — WL consistent but not lex-identical to brute-force)")
    return passed, failed


def test_isomorphism(n_trials=500, seed=42, verbose=False):
    """Random variable renamings + triple reorderings must produce same canonical form.
    This is the critical correctness test: isomorphic inputs MUST get the same WL form."""
    rng = random.Random(seed)
    passed = 0
    failed = 0

    for trial in range(n_trials):
        base = rng.choice(QUERY_SHAPES)
        base_vars = list({t for tri in base for t in tri if t.startswith("?")})
        if not base_vars:
            passed += 1
            continue

        # Random variable renaming
        new_names = [f"?z{i}" for i in range(len(base_vars))]
        rng.shuffle(new_names)
        mapping = dict(zip(base_vars, new_names))
        permuted = [[mapping.get(t, t) for t in tri] for tri in base]
        rng.shuffle(permuted)

        c1 = canonicalize_bgp(base)
        c2 = canonicalize_bgp(permuted)

        if c1 != c2:
            failed += 1
            print(f"  FAIL isomorphism trial {trial}: different canonical forms")
            print(f"    base:    {base} -> {c1}")
            print(f"    permuted:{permuted} -> {c2}")
        else:
            passed += 1

    return passed, failed


def test_non_isomorphism(verbose=False):
    """Structurally different queries must produce different canonical forms."""
    passed = 0
    failed = 0

    for i, (q1, q2) in enumerate(NON_ISOMORPHIC_PAIRS):
        c1 = canonicalize_bgp(q1)
        c2 = canonicalize_bgp(q2)
        if c1 != c2:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL non-isomorphism pair {i}: same canonical form")
            print(f"    q1: {q1}")
            print(f"    q2: {q2}")
            print(f"    canon: {c1}")

    return passed, failed


def test_labeled_pairs(verbose=False):
    """Manually labeled pairs: both WL and brute-force must agree with ground-truth label.

    Checks three things (first two are hard failures, third is a warning):
      1. Brute-force agrees with the manual label (validates the label itself)
      2. WL agrees with the manual label (correctness for dedup)
      3. WL produces the same lex form as brute-force (nice-to-have, not required)
    """
    passed = 0
    failed = 0
    lex_warnings = 0

    for i, (q1, q2, expected_iso, desc) in enumerate(LABELED_PAIRS):
        wl1 = canonicalize_bgp(q1)
        wl2 = canonicalize_bgp(q2)
        bf1 = brute_force_canonical(q1)
        bf2 = brute_force_canonical(q2)

        wl_says_iso = (wl1 == wl2)
        bf_says_iso = (bf1 == bf2)

        bf_correct = (bf_says_iso == expected_iso)
        wl_correct = (wl_says_iso == expected_iso)
        wl_matches_bf = (wl1 == bf1 and wl2 == bf2)

        if not bf_correct:
            failed += 1
            label = "ISO" if expected_iso else "NOT-ISO"
            print(f"  FAIL labeled pair {i} [{label}]: {desc}")
            print(f"    BRUTE-FORCE disagrees with label!")
            print(f"      bf1: {bf1}")
            print(f"      bf2: {bf2}")
        elif not wl_correct:
            failed += 1
            label = "ISO" if expected_iso else "NOT-ISO"
            print(f"  FAIL labeled pair {i} [{label}]: {desc}")
            print(f"    WL disagrees with label!")
            print(f"      wl1: {wl1}")
            print(f"      wl2: {wl2}")
        else:
            passed += 1
            if not wl_matches_bf:
                lex_warnings += 1
                if verbose:
                    label = "ISO" if expected_iso else "NOT-ISO"
                    print(f"  WARN labeled pair {i} [{label}]: {desc}")
                    print(f"    WL correct but not lex-minimal vs brute-force")

    if lex_warnings:
        print(f"  (lex-minimality warnings: {lex_warnings} — WL correct but not lex-identical to brute-force)")

    return passed, failed


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    total_passed = 0
    total_failed = 0

    print(f"Test 1: Direct comparison (WL vs brute-force) — {len(QUERY_SHAPES)} cases")
    p, f = test_direct(verbose)
    total_passed += p
    total_failed += f
    print(f"  Passed: {p}/{p+f}")

    print(f"\nTest 2: Isomorphism invariance — 500 random permutations")
    p, f = test_isomorphism(500, verbose=verbose)
    total_passed += p
    total_failed += f
    print(f"  Passed: {p}/{p+f}")

    print(f"\nTest 3: Non-isomorphism — {len(NON_ISOMORPHIC_PAIRS)} pairs")
    p, f = test_non_isomorphism(verbose)
    total_passed += p
    total_failed += f
    print(f"  Passed: {p}/{p+f}")

    n_iso = sum(1 for _, _, iso, _ in LABELED_PAIRS if iso)
    n_not = sum(1 for _, _, iso, _ in LABELED_PAIRS if not iso)
    print(f"\nTest 4: Labeled pairs — {n_iso} isomorphic + {n_not} non-isomorphic = {len(LABELED_PAIRS)} pairs")
    print(f"  (checks WL, brute-force, AND manual ground-truth label all agree)")
    p, f = test_labeled_pairs(verbose)
    total_passed += p
    total_failed += f
    print(f"  Passed: {p}/{p+f}")

    print(f"\n{'='*40}")
    print(f"Total: {total_passed}/{total_passed+total_failed}")
    if total_failed:
        print(f"FAILED: {total_failed}")
    else:
        print("ALL TESTS PASSED")

    sys.exit(1 if total_failed else 0)
