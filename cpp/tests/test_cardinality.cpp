#include "doctest.h"

#include "bgpgen/cardinality.hpp"
#include "bgpgen/nt_parser.hpp"

#include <algorithm>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using namespace bgpgen;

namespace {

struct Fixture {
  TermTable terms;
  TripleStore store;
  SubjectIndex fwd, rev;
  std::vector<Triple> data; // deduplicated, frozen ids — brute-force ground truth
};

std::unique_ptr<Fixture> load(const std::string& nt) {
  auto out = std::make_unique<Fixture>();
  std::vector<Triple> triples = parse_nt_buffer(nt.data(), nt.size(), out->terms, 0);
  std::vector<Triple>* sets[] = {&triples};
  out->terms.freeze(sets);
  out->data = triples;
  auto lt = [](const Triple& a, const Triple& b) {
    return std::tie(a.s, a.p, a.o) < std::tie(b.s, b.p, b.o);
  };
  auto eq = [](const Triple& a, const Triple& b) {
    return a.s == b.s && a.p == b.p && a.o == b.o;
  };
  std::sort(out->data.begin(), out->data.end(), lt);
  out->data.erase(std::unique(out->data.begin(), out->data.end(), eq), out->data.end());
  out->store.build(std::move(triples), 1);
  out->fwd.build(out->store, out->terms.bnode_base());
  out->rev.build(out->store, out->terms.bnode_base(), true);
  return out;
}

/// Ground truth: nested-loop homomorphism count. Every variable (predicate
/// vars included) ranges over all terms; each pattern must match some triple.
uint64_t brute(const std::vector<Triple>& data, const std::vector<TriplePattern>& pats,
               std::vector<TermId>& assign, size_t i) {
  if (i == pats.size()) return 1;
  const TriplePattern& tp = pats[i];
  uint64_t total = 0;
  for (const Triple& t : data) {
    uint32_t newly[3];
    size_t n_new = 0;
    bool ok = true;
    QTerm qs[3] = {tp.s, tp.p, tp.o};
    TermId vals[3] = {t.s, t.p, t.o};
    for (int k = 0; k < 3 && ok; ++k) {
      if (!q_is_var(qs[k])) {
        ok = qs[k] == vals[k];
      } else {
        uint32_t v = q_var_id(qs[k]);
        if (assign[v] == kInvalidTerm) {
          assign[v] = vals[k];
          newly[n_new++] = v;
        } else {
          ok = assign[v] == vals[k];
        }
      }
    }
    if (ok) total += brute(data, pats, assign, i + 1);
    for (size_t k = 0; k < n_new; ++k) assign[newly[k]] = kInvalidTerm;
  }
  return total;
}

uint64_t brute_count(const Fixture& g, const std::vector<TriplePattern>& pats) {
  uint32_t n_vars = 0;
  for (const TriplePattern& tp : pats)
    for (QTerm t : {tp.s, tp.p, tp.o})
      if (q_is_var(t)) n_vars = std::max(n_vars, q_var_id(t) + 1);
  std::vector<TermId> assign(n_vars, kInvalidTerm);
  return brute(g.data, pats, assign, 0);
}

/// Builds patterns from tokens: "?name" = variable, anything else = IRI
/// looked up in the term table (must exist in the data).
struct QueryBuilder {
  const TermTable& terms;
  std::unordered_map<std::string, uint32_t> vars;

  Query operator()(std::initializer_list<std::array<const char*, 3>> pats) {
    Query q;
    for (const auto& t : pats) q.patterns.push_back({term(t[0]), term(t[1]), term(t[2])});
    q.var_names.resize(vars.size());
    for (const auto& [name, id] : vars) q.var_names[id] = name;
    return q;
  }

  QTerm term(const std::string& tok) {
    if (tok[0] == '?') {
      auto [it, ins] = vars.try_emplace(tok, static_cast<uint32_t>(vars.size()));
      return qvar(it->second);
    }
    TermId id = terms.find_iri(tok);
    REQUIRE(id != kInvalidTerm);
    return id;
  }
};

int64_t count(const Fixture& g, const Query& q, uint64_t budget = 50'000'000) {
  return count_query(g.store, g.fwd, g.rev, q, budget);
}

void check_vs_brute(const Fixture& g, const Query& q) {
  CHECK(count(g, q) == static_cast<int64_t>(brute_count(g, q.patterns)));
}

// Small graph exercising hubs, shared objects, a self-loop, a cycle, and
// literal objects.
const char* kGraph =
    "<http://a> <http://p> <http://b> .\n"
    "<http://a> <http://p> <http://c> .\n"
    "<http://a> <http://q> <http://d> .\n"
    "<http://b> <http://p> <http://c> .\n"
    "<http://b> <http://q> <http://d> .\n"
    "<http://c> <http://p> <http://a> .\n" // closes a triangle a->b->c->a? (c->a)
    "<http://c> <http://q> <http://c> .\n" // self-loop
    "<http://d> <http://p> <http://d> .\n" // self-loop on p
    "<http://a> <http://r> \"lit one\" .\n"
    "<http://b> <http://r> \"lit one\" .\n"
    "<http://b> <http://r> \"lit two\"@en .\n"
    "<http://e> <http://q> <http://a> .\n"
    "<http://e> <http://q> <http://b> .\n";

} // namespace

TEST_CASE("hand-picked queries match the brute-force homomorphism count") {
  auto g = load(kGraph);
  auto q = [&](std::initializer_list<std::array<const char*, 3>> pats) {
    return QueryBuilder{g->terms, {}}(pats);
  };

  // Single patterns: bound/var in every combination.
  check_vs_brute(*g, q({{"?s", "http://p", "?o"}}));
  check_vs_brute(*g, q({{"http://a", "http://p", "?o"}}));
  check_vs_brute(*g, q({{"?s", "http://p", "http://c"}}));
  check_vs_brute(*g, q({{"http://a", "http://p", "http://b"}}));
  check_vs_brute(*g, q({{"?s", "?p", "?o"}}));      // total triples
  check_vs_brute(*g, q({{"http://b", "?p", "?o"}})); // out-degree
  check_vs_brute(*g, q({{"?s", "?p", "http://d"}})); // in-degree
  check_vs_brute(*g, q({{"http://a", "?p", "http://c"}}));

  // Stars (hub join), incl. same predicate twice and a constant arm.
  check_vs_brute(*g, q({{"?s", "http://p", "?a"}, {"?s", "http://q", "?b"}}));
  check_vs_brute(*g, q({{"?s", "http://p", "?a"}, {"?s", "http://p", "?b"}}));
  check_vs_brute(*g, q({{"?s", "http://p", "?a"}, {"?s", "http://q", "http://d"}}));
  check_vs_brute(*g, q({{"?s", "http://p", "?a"}, {"?s", "?p1", "?b"}}));

  // Duplicate pattern: must count like a single occurrence.
  check_vs_brute(*g, q({{"?s", "http://p", "?a"}, {"?s", "http://p", "?a"}}));

  // Paths and a cycle.
  check_vs_brute(*g, q({{"?a", "http://p", "?b"}, {"?b", "http://p", "?c"}}));
  check_vs_brute(*g, q({{"?a", "http://p", "?b"}, {"?b", "http://q", "?c"}}));
  check_vs_brute(*g,
                 q({{"http://a", "http://p", "?b"}, {"?b", "http://p", "?c"}}));
  check_vs_brute(
      *g, q({{"?a", "http://p", "?b"}, {"?b", "http://p", "?c"}, {"?c", "http://p", "?a"}}));

  // Self-loop pattern.
  check_vs_brute(*g, q({{"?x", "http://p", "?x"}}));
  check_vs_brute(*g, q({{"?x", "?p", "?x"}}));

  // In-star (object join) and mixed directions.
  check_vs_brute(*g, q({{"?a", "http://q", "?s"}, {"?b", "http://q", "?s"}}));
  check_vs_brute(*g, q({{"?a", "http://p", "?j"}, {"?j", "http://q", "?b"}}));

  // Variable predicates joining through nodes.
  check_vs_brute(*g, q({{"?a", "?p", "?b"}, {"?b", "?q", "?c"}}));

  // Disconnected: cartesian product of two factors.
  check_vs_brute(*g, q({{"?a", "http://p", "?b"}, {"?c", "http://q", "?d"}}));

  // Literal-object arm reached through a join.
  check_vs_brute(*g, q({{"?s", "http://r", "?lit"}, {"?s", "http://p", "?o"}}));
}

TEST_CASE("zero matches and absent predicates count 0") {
  auto g = load(kGraph);
  QueryBuilder qb{g->terms, {}};
  // e has no outgoing p edges.
  Query q1 = qb({{"http://e", "http://p", "?o"}});
  CHECK(count(*g, q1) == 0);
  // A predicate IRI that exists as a term but not as a predicate.
  Query q2;
  q2.patterns.push_back({qvar(0), g->terms.find_iri("http://a"), qvar(1)});
  q2.var_names = {"?s", "?o"};
  CHECK(count(*g, q2) == 0);
}

TEST_CASE("budget exhaustion yields -1") {
  auto g = load(kGraph);
  QueryBuilder qb{g->terms, {}};
  Query q = qb({{"?a", "?p", "?b"}, {"?b", "?q", "?c"}});
  CHECK(count(*g, q, 2) == -1);
  CHECK(count(*g, q) > 0); // sanity: full budget succeeds
}

TEST_CASE("random queries match brute force") {
  auto g = load(kGraph);
  std::mt19937_64 rng(7);
  const char* var_pool[] = {"?v0", "?v1", "?v2", "?v3"};

  for (int trial = 0; trial < 300; ++trial) {
    CAPTURE(trial);
    size_t n_pats = 1 + rng() % 3;
    Query q;
    std::unordered_map<std::string, uint32_t> vars;
    auto var_term = [&](const char* name) {
      auto [it, ins] = vars.try_emplace(name, static_cast<uint32_t>(vars.size()));
      return qvar(it->second);
    };
    for (size_t i = 0; i < n_pats; ++i) {
      const Triple& t = g->data[rng() % g->data.size()];
      TriplePattern tp{t.s, t.p, t.o};
      // Each position independently: keep the constant or swap in a var
      // from a small pool so joins (and self-joins) happen often.
      if (rng() % 4 != 0) tp.s = var_term(var_pool[rng() % 4]);
      if (rng() % 2 != 0) tp.p = var_term(var_pool[rng() % 4]);
      if (rng() % 4 != 0) tp.o = var_term(var_pool[rng() % 4]);
      q.patterns.push_back(tp);
    }
    q.var_names.resize(vars.size());
    for (const auto& [name, id] : vars) q.var_names[id] = name;
    CHECK(count(*g, q) == static_cast<int64_t>(brute_count(*g, q.patterns)));
  }
}

TEST_CASE("parallel count_cardinalities fills every y") {
  auto g = load(kGraph);
  QueryBuilder qb{g->terms, {}};
  std::vector<Query> qs;
  qs.push_back(qb({{"?s", "http://p", "?o"}}));
  qs.push_back(qb({{"?s", "http://p", "?a"}, {"?s", "http://q", "?b"}}));
  qs.push_back(qb({{"http://e", "http://p", "?o"}}));
  count_cardinalities(g->store, g->fwd, g->rev, qs, {});
  for (const Query& q : qs) CHECK(q.y == static_cast<int64_t>(brute_count(*g, q.patterns)));
}
