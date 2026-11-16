#include "doctest.h"

#include "bgpgen/dedup.hpp"

#include <algorithm>
#include <array>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using namespace bgpgen;

namespace {

// Test data ported from test_canonicalization.py. Queries are written as
// token triples; the Builder fabricates TermIds for constants (shared across
// a pair so equal tokens compare equal) and dense var ids.

using Tok = std::array<const char*, 3>;
using Q = std::vector<Tok>;

struct Builder {
  std::unordered_map<std::string, uint32_t> vars;
  std::unordered_map<std::string, TermId> consts;

  std::vector<TriplePattern> operator()(const Q& q) {
    std::vector<TriplePattern> out;
    for (const Tok& t : q) out.push_back({term(t[0]), term(t[1]), term(t[2])});
    return out;
  }

  QTerm term(const std::string& tok) {
    if (tok[0] == '?') {
      auto [it, ins] = vars.try_emplace(tok, static_cast<uint32_t>(vars.size()));
      return qvar(it->second);
    }
    auto [it, ins] = consts.try_emplace(tok, static_cast<TermId>(100 + consts.size()));
    return it->second;
  }
};

uint32_t n_vars(const std::vector<TriplePattern>& ps) {
  uint32_t n = 0;
  for (const TriplePattern& tp : ps)
    for (QTerm t : {tp.s, tp.p, tp.o})
      if (q_is_var(t)) n = std::max(n, q_var_id(t) + 1);
  return n;
}

std::vector<TriplePattern> rename(std::vector<TriplePattern> ps,
                                  const std::vector<uint32_t>& perm) {
  auto re = [&](QTerm& t) {
    if (q_is_var(t)) t = qvar(perm[q_var_id(t)]);
  };
  for (TriplePattern& tp : ps) {
    re(tp.s);
    re(tp.p);
    re(tp.o);
  }
  return ps;
}

// QUERY_SHAPES: stars, paths, cycles, diamond, trees, self-joins, mixed
// bound/unbound, variable predicates, no variables.
const std::vector<Q> kShapes = {
    {{"?s", "<p1>", "?a"}, {"?s", "<p2>", "?b"}},
    {{"?s", "<p>", "?a"}, {"?s", "<p>", "?b"}},
    {{"?s", "<p>", "?a"}, {"?s", "<p>", "?b"}, {"?s", "<p>", "?c"}},
    {{"?s", "<p1>", "<obj1>"}, {"?s", "<p2>", "?o"}},
    {{"?s", "<p1>", "?a"}, {"?s", "<p2>", "?b"}, {"?s", "<p3>", "?c"}, {"?s", "<p4>", "?d"}},
    {{"?s", "<p>", "?a"}, {"?s", "<p>", "?b"}, {"?s", "<p>", "?c"}, {"?s", "<p>", "?d"}},
    {{"?a", "<p1>", "?b"}, {"?b", "<p2>", "?c"}},
    {{"?a", "<p1>", "?b"}, {"?b", "<p2>", "?c"}, {"?c", "<p3>", "?d"}},
    {{"<X>", "<p1>", "?b"}, {"?b", "<p2>", "?c"}, {"?c", "<p3>", "<Y>"}},
    {{"<X>", "<p1>", "?b"}, {"?b", "<p2>", "?c"}, {"?c", "<p3>", "?d"}, {"?d", "<p4>", "<Y>"}},
    {{"?a", "<p1>", "?b"}, {"?b", "<p2>", "?c"}, {"?c", "<p3>", "?a"}},
    {{"?a", "<p>", "?b"}, {"?b", "<p>", "?c"}, {"?c", "<p>", "?a"}},
    {{"?a", "<p1>", "?b"}, {"?b", "<p2>", "?d"}, {"?a", "<p3>", "?c"}, {"?c", "<p4>", "?d"}},
    {{"?r", "<p1>", "?a"}, {"?r", "<p2>", "?b"}, {"?a", "<p3>", "?c"}, {"?a", "<p4>", "?d"}},
    {{"?r", "<p1>", "?a"}, {"?r", "<p2>", "?b"}, {"?b", "<p3>", "?c"}},
    {{"?x", "<p>", "?x"}},
    {{"?x", "<p1>", "?y"}, {"?y", "<p2>", "?x"}},
    {{"<A>", "<p1>", "?b"}, {"?b", "<p2>", "?c"}},
    {{"?a", "<p1>", "?b"}, {"?b", "<p2>", "<C>"}},
    {{"?s", "?p", "?o1"}, {"?s", "<p2>", "?o2"}},
    {{"?a", "?p", "?b"}, {"?b", "?q", "?c"}},
    {{"<A>", "<p>", "<B>"}},
    {{"<A>", "<p1>", "<B>"}, {"<B>", "<p2>", "<C>"}},
};

struct Pair {
  Q a, b;
};

const std::vector<Pair> kNonIsoPairs = {
    // Different predicates on path
    {{{"?x", "<p1>", "?y"}, {"?y", "<p2>", "?z"}},
     {{"?x", "<p2>", "?y"}, {"?y", "<p1>", "?z"}}},
    // Star vs path
    {{{"?s", "<p1>", "?a"}, {"?s", "<p2>", "?b"}},
     {{"?a", "<p1>", "?b"}, {"?b", "<p2>", "?c"}}},
    // Different bound objects
    {{{"?s", "<p1>", "<obj1>"}, {"?s", "<p2>", "?o"}},
     {{"?s", "<p1>", "<obj2>"}, {"?s", "<p2>", "?o"}}},
    // Bound vs unbound
    {{{"?s", "<p1>", "<obj1>"}, {"?s", "<p2>", "?o"}},
     {{"?s", "<p1>", "?a"}, {"?s", "<p2>", "?b"}}},
    // Different size
    {{{"?s", "<p1>", "?a"}, {"?s", "<p2>", "?b"}},
     {{"?s", "<p1>", "?a"}, {"?s", "<p2>", "?b"}, {"?s", "<p3>", "?c"}}},
    // Triangle vs open path
    {{{"?a", "<p1>", "?b"}, {"?b", "<p2>", "?c"}, {"?c", "<p3>", "?a"}},
     {{"?a", "<p1>", "?b"}, {"?b", "<p2>", "?c"}, {"?c", "<p3>", "?d"}}},
    // Self-join vs no self-join
    {{{"?x", "<p>", "?x"}}, {{"?x", "<p>", "?y"}}},
    // Different join structure
    {{{"?a", "<p1>", "?b"}, {"?c", "<p2>", "?b"}},
     {{"?a", "<p1>", "?b"}, {"?b", "<p2>", "?c"}}},
    // Diamond: different predicate assignment
    {{{"?a", "<p1>", "?b"}, {"?b", "<p2>", "?d"}, {"?a", "<p3>", "?c"}, {"?c", "<p4>", "?d"}},
     {{"?a", "<p1>", "?b"}, {"?b", "<p3>", "?d"}, {"?a", "<p2>", "?c"}, {"?c", "<p4>", "?d"}}},
};

struct LabeledPair {
  Q a, b;
  bool iso;
  const char* desc;
};

const std::vector<LabeledPair> kLabeledPairs = {
    // -- isomorphic --
    {{{"?a", "<p1>", "?b"}, {"?b", "<p2>", "?c"}},
     {{"?x", "<p1>", "?y"}, {"?y", "<p2>", "?z"}},
     true,
     "path: simple variable renaming"},
    {{{"?s", "<p1>", "?a"}, {"?s", "<p2>", "?b"}, {"?s", "<p3>", "?c"}},
     {{"?center", "<p3>", "?z"}, {"?center", "<p1>", "?x"}, {"?center", "<p2>", "?y"}},
     true,
     "star: reordered triples + renamed vars"},
    {{{"?s", "<p>", "?a"}, {"?s", "<p>", "?b"}, {"?s", "<p>", "?c"}},
     {{"?hub", "<p>", "?x"}, {"?hub", "<p>", "?y"}, {"?hub", "<p>", "?z"}},
     true,
     "star: all same predicate, interchangeable leaves"},
    {{{"?a", "<p>", "?b"}, {"?b", "<p>", "?c"}, {"?c", "<p>", "?a"}},
     {{"?x", "<p>", "?y"}, {"?y", "<p>", "?z"}, {"?z", "<p>", "?x"}},
     true,
     "cycle: all same predicate, full symmetry (needs individualization)"},
    {{{"?a", "<p>", "?b"}, {"?a", "<p>", "?c"}, {"?b", "<p>", "?d"}, {"?c", "<p>", "?d"}},
     {{"?w", "<p>", "?x"}, {"?w", "<p>", "?y"}, {"?x", "<p>", "?z"}, {"?y", "<p>", "?z"}},
     true,
     "diamond: all same predicate, high symmetry"},
    {{{"?x", "<p>", "?x"}}, {{"?y", "<p>", "?y"}}, true, "self-loop renaming"},
    {{{"?x", "<p1>", "?y"}, {"?y", "<p2>", "?x"}},
     {{"?b", "<p1>", "?a"}, {"?a", "<p2>", "?b"}},
     true,
     "mutual edges: swap roles"},
    {{{"?x", "<p>", "?y"}, {"?y", "<p>", "?x"}},
     {{"?b", "<p>", "?a"}, {"?a", "<p>", "?b"}},
     true,
     "mutual edges, same predicate: swap roles"},
    {{{"<A>", "<p1>", "?x"}, {"?x", "<p2>", "?y"}, {"?y", "<p2>", "<B>"}},
     {{"<A>", "<p1>", "?m"}, {"?m", "<p2>", "?n"}, {"?n", "<p2>", "<B>"}},
     true,
     "path: bound endpoints, rename interior vars"},
    {{{"?s", "?p", "?o1"}, {"?s", "?p", "?o2"}},
     {{"?x", "?r", "?y"}, {"?x", "?r", "?z"}},
     true,
     "variable predicates: star with shared var predicate"},
    {{{"?a", "<p>", "?b"}, {"?b", "<p>", "?c"}, {"?c", "<p>", "?d"}},
     {{"?w", "<p>", "?x"}, {"?x", "<p>", "?y"}, {"?y", "<p>", "?z"}},
     true,
     "long same-predicate path (interior nodes tie under WL-1)"},
    {{{"?a", "?p", "?b"}, {"?b", "?p", "?c"}, {"?c", "?p", "?d"}},
     {{"?w", "?p", "?x"}, {"?x", "?p", "?y"}, {"?y", "?p", "?z"}},
     true,
     "long var-predicate path (interior nodes tie under WL-1)"},
    {{{"?r", "<p1>", "?a"}, {"?r", "<p2>", "?b"}, {"?b", "<p3>", "?c"}},
     {{"?root", "<p1>", "?x"}, {"?root", "<p2>", "?y"}, {"?y", "<p3>", "?z"}},
     true,
     "asymmetric tree: rename all vars"},
    {{{"?r", "<p>", "?a"}, {"?r", "<p>", "?b"}, {"?a", "<p>", "?c"}, {"?a", "<p>", "?d"}},
     {{"?root", "<p>", "?l"}, {"?root", "<p>", "?r2"}, {"?l", "<p>", "?ll"}, {"?l", "<p>", "?lr"}},
     true,
     "binary tree: same predicates, partial symmetry"},
    {{{"?a", "<p>", "?b"}, {"?b", "<p>", "?c"}, {"?c", "<p>", "?d"}, {"?d", "<p>", "?a"}},
     {{"?w", "<p>", "?x"}, {"?x", "<p>", "?y"}, {"?y", "<p>", "?z"}, {"?z", "<p>", "?w"}},
     true,
     "4-cycle: all same predicate (dihedral symmetry)"},
    {{{"?c", "<p1>", "?a"}, {"?c", "<p2>", "?b"}, {"?b", "<p3>", "?d"}},
     {{"?hub", "<p1>", "?x"}, {"?hub", "<p2>", "?y"}, {"?y", "<p3>", "?z"}},
     true,
     "flower: star center with one branch extending"},
    {{{"?a", "<p1>", "?b"}, {"?c", "<p2>", "?d"}},
     {{"?x", "<p1>", "?y"}, {"?w", "<p2>", "?z"}},
     true,
     "disconnected: two independent edges"},
    {{{"?c", "<p>", "?a"},
      {"?c", "<p>", "?b"},
      {"?a", "<p>", "?a1"},
      {"?a", "<p>", "?a2"},
      {"?b", "<p>", "?b1"},
      {"?b", "<p>", "?b2"}},
     {{"?hub", "<p>", "?l"},
      {"?hub", "<p>", "?r"},
      {"?l", "<p>", "?ll"},
      {"?l", "<p>", "?lr"},
      {"?r", "<p>", "?rl"},
      {"?r", "<p>", "?rr"}},
     true,
     "snowflake: full symmetry, 7 vars, WL-1 ties everywhere"},

    // -- non-isomorphic --
    {{{"?a", "<p1>", "?b"}, {"?b", "<p2>", "?c"}},
     {{"?a", "<p2>", "?b"}, {"?b", "<p1>", "?c"}},
     false,
     "path: reversed predicate order"},
    {{{"?s", "<p1>", "?a"}, {"?s", "<p2>", "?b"}},
     {{"?a", "<p1>", "?b"}, {"?b", "<p2>", "?c"}},
     false,
     "star vs path (different topology)"},
    {{{"?x", "<p>", "?x"}}, {{"?x", "<p>", "?y"}}, false, "self-loop vs distinct endpoints"},
    {{{"?a", "<p1>", "?b"}, {"?b", "<p2>", "?c"}, {"?c", "<p3>", "?a"}},
     {{"?a", "<p1>", "?b"}, {"?b", "<p2>", "?c"}, {"?c", "<p3>", "?d"}},
     false,
     "triangle vs open path"},
    {{{"?s", "<p>", "<A>"}}, {{"?s", "<p>", "<B>"}}, false, "different bound object"},
    {{{"?a", "<p1>", "?b"}, {"?b", "<p2>", "?d"}, {"?a", "<p3>", "?c"}, {"?c", "<p4>", "?d"}},
     {{"?a", "<p1>", "?b"}, {"?b", "<p3>", "?d"}, {"?a", "<p2>", "?c"}, {"?c", "<p4>", "?d"}},
     false,
     "diamond: swapped predicates on different arms"},
    {{{"?a", "<p1>", "?s"}, {"?b", "<p2>", "?s"}},
     {{"?s", "<p1>", "?a"}, {"?s", "<p2>", "?b"}},
     false,
     "in-star vs out-star (different join positions)"},
    {{{"?a", "<p1>", "?b"}, {"?c", "<p2>", "?d"}},
     {{"?a", "<p1>", "?b"}, {"?b", "<p2>", "?c"}},
     false,
     "disconnected pair vs connected path"},
    {{{"?s", "<p>", "?a"}, {"?s", "<p>", "?b"}, {"?s", "<p>", "?c"}},
     {{"?a", "<p>", "?b"}, {"?b", "<p>", "?c"}, {"?c", "<p>", "?d"}},
     false,
     "same-pred star-3 vs path-3 (same multiset, different structure)"},
    {{{"?a", "<p>", "?b"}, {"?b", "<p>", "?c"}, {"?c", "<p>", "?d"}},
     {{"?a", "<p>", "?b"}, {"?b", "<p>", "?c"}, {"?c", "<p>", "?a"}},
     false,
     "same-pred path-3 vs triangle (same multiset, acyclic vs cyclic)"},
    {{{"?a", "<p>", "?b"}, {"?b", "<p>", "?c"}, {"?c", "<p>", "?d"}, {"?d", "<p>", "?a"}},
     {{"?a", "<p>", "?b"}, {"?b", "<p>", "?a"}, {"?c", "<p>", "?d"}, {"?d", "<p>", "?c"}},
     false,
     "4-cycle vs two mutual edges (same degree sequence)"},
    {{{"?r", "<p>", "?a"}, {"?r", "<p>", "?b"}, {"?a", "<p>", "?c"}, {"?b", "<p>", "?d"}},
     {{"?r", "<p>", "?a"}, {"?r", "<p>", "?b"}, {"?a", "<p>", "?c"}, {"?a", "<p>", "?d"}},
     false,
     "balanced vs unbalanced tree (same-pred, same degree sequence)"},
    {{{"?a", "?p", "?b"}, {"?b", "<p2>", "?c"}},
     {{"?a", "<p2>", "?b"}, {"?b", "?p", "?c"}},
     false,
     "variable predicate on first vs second triple"},
    {{{"<A>", "<p1>", "?x"}, {"?x", "<p2>", "?y"}},
     {{"?x", "<p1>", "?y"}, {"?y", "<p2>", "<A>"}},
     false,
     "bound at start vs bound at end of path"},
    {{{"?s", "<p>", "?a"}, {"?s", "<p>", "?b"}, {"?s", "<p>", "?c"}},
     {{"?s", "<p>", "?a"}, {"?s", "<p>", "?b"}, {"?s", "<p>", "?c"}, {"?s", "<p>", "?d"}},
     false,
     "same-pred star: 3 vs 4 leaves"},
};

} // namespace

TEST_CASE("canonical key invariant under every variable permutation") {
  for (size_t i = 0; i < kShapes.size(); ++i) {
    CAPTURE(i);
    Builder build;
    std::vector<TriplePattern> base = build(kShapes[i]);
    const std::string key = canonical_bgp_key(base);
    std::vector<uint32_t> perm(n_vars(base));
    std::iota(perm.begin(), perm.end(), 0u);
    do {
      std::vector<TriplePattern> renamed = rename(base, perm);
      CHECK(canonical_bgp_key(renamed) == key);
      std::reverse(renamed.begin(), renamed.end());
      CHECK(canonical_bgp_key(renamed) == key);
    } while (std::next_permutation(perm.begin(), perm.end()));
  }
}

TEST_CASE("canonical key invariant under random renaming + reordering") {
  std::mt19937_64 rng(42);
  for (int trial = 0; trial < 500; ++trial) {
    size_t i = rng() % kShapes.size();
    CAPTURE(trial);
    CAPTURE(i);
    Builder build;
    std::vector<TriplePattern> base = build(kShapes[i]);
    std::vector<uint32_t> perm(n_vars(base));
    std::iota(perm.begin(), perm.end(), 0u);
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<TriplePattern> renamed = rename(base, perm);
    std::shuffle(renamed.begin(), renamed.end(), rng);
    CHECK(canonical_bgp_key(renamed) == canonical_bgp_key(base));
  }
}

TEST_CASE("non-isomorphic pairs get different keys") {
  for (size_t i = 0; i < kNonIsoPairs.size(); ++i) {
    CAPTURE(i);
    Builder build; // shared so equal constant tokens get equal TermIds
    CHECK(canonical_bgp_key(build(kNonIsoPairs[i].a)) !=
          canonical_bgp_key(build(kNonIsoPairs[i].b)));
  }
}

TEST_CASE("labeled pairs match ground-truth isomorphism") {
  for (const LabeledPair& lp : kLabeledPairs) {
    CAPTURE(lp.desc);
    Builder build;
    bool same = canonical_bgp_key(build(lp.a)) == canonical_bgp_key(build(lp.b));
    CHECK(same == lp.iso);
  }
}

TEST_CASE("Dedup wl collapses isomorphic queries, keeps distinct ones") {
  TermTable terms; // unused by the wl path
  Dedup dedup(DedupMethod::Wl);
  Builder build;

  Query star;
  star.patterns = build({{"?s", "<p>", "?a"}, {"?s", "<p>", "?b"}});
  CHECK(dedup.insert(star, terms));

  Query renamed;
  renamed.patterns = build({{"?hub", "<p>", "?y"}, {"?hub", "<p>", "?x"}});
  CHECK(!dedup.insert(renamed, terms)); // isomorphic to star

  Query path;
  path.patterns = build({{"?a", "<p>", "?b"}, {"?b", "<p>", "?c"}});
  CHECK(dedup.insert(path, terms)); // same predicate multiset, new structure

  CHECK(dedup.size() == 2);
}

namespace {

/// A random query of the template: predicates from two constants or a
/// variable, slots bound to one of two constants of their own (so different
/// slots never share a constant) or left variable. With p_split > 0 a bound
/// slot is written as its variable in some of its triples.
Query random_template_query(const CompiledTemplate& t, std::mt19937_64& rng, double p_split) {
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::vector<QTerm> slot_term(t.n_slots);
  for (uint8_t s = 0; s < t.n_slots; ++s)
    slot_term[s] = unit(rng) < 0.3 ? static_cast<QTerm>(300 + 2 * s + rng() % 2) : qvar(s);
  Query q;
  for (size_t e = 0; e < t.triples.size(); ++e) {
    auto node = [&](uint8_t slot) {
      return unit(rng) < p_split ? qvar(slot) : slot_term[slot];
    };
    QTerm p = unit(rng) < 0.2 ? qvar(static_cast<uint32_t>(t.n_slots + e))
                              : static_cast<QTerm>(200 + rng() % 2);
    q.patterns.push_back({node(t.triples[e].s_slot), p, node(t.triples[e].o_slot)});
  }
  return q;
}

} // namespace

TEST_CASE("Dedup symmetry makes the same decisions as wl") {
  const char* shapes[] = {
      "?r B ?a, ?r B ?b, ?r B ?c",
      "?a B ?b, ?b B ?c, ?c B ?d",
      "?a B ?b, ?b B ?c, ?c B ?a",
      "?r B ?a, ?r B ?b, ?a B ?c, ?b B ?d",
      "?a B ?m, ?b B ?m, ?m B ?c, ?m B ?d",
      "?v0 B ?v1, ?v0 B ?v4, ?v2 B ?v4, ?v3 B ?v4, ?v4 B ?v5, ?v4 B ?v6",
      "?a B ?b, ?b B ?c, ?c B ?d, ?d B ?a", // cycle of four
      "?v0 B ?v1, ?v1 B ?v2, ?v1 B ?v3, ?v1 B ?v4, ?v1 B ?v5", // CEG-like star, one edge in
      // two groups of interchangeable leaves that a symmetry swaps
      "?m B ?h, ?m B ?k, ?h B ?a, ?h B ?b, ?k B ?c, ?k B ?d",
      "?a B ?h, ?b B ?h, ?h B ?k, ?k B ?c, ?k B ?d",
  };
  TermTable terms;
  std::mt19937_64 rng(1);
  for (const char* shape : shapes) {
    CompiledTemplate t = parse_template(shape);
    for (double p_split : {0.0, 0.3}) {
      Dedup sym(DedupMethod::Symmetry), wl(DedupMethod::Wl);
      sym.set_template(t);
      size_t kept = 0;
      for (int i = 0; i < 5000; ++i) {
        Query q = random_template_query(t, rng, p_split);
        bool a = sym.insert(q, terms), b = wl.insert(q, terms);
        REQUIRE(a == b);
        kept += a;
      }
      CHECK(kept > 10);
      CHECK(kept < 5000);
    }
  }
}

TEST_CASE("Dedup symmetry without a template falls back to wl") {
  Builder b;
  TermTable terms;
  Dedup sym(DedupMethod::Symmetry);
  Query q1, q2;
  q1.patterns = b({{"?s", "<p1>", "?a"}, {"?s", "<p2>", "?b"}});
  q2.patterns = b({{"?x", "<p2>", "?u"}, {"?x", "<p1>", "?w"}});
  CHECK(sym.insert(q1, terms));
  CHECK(!sym.insert(q2, terms));
}
