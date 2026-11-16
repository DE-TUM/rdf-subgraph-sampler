#include "doctest.h"

#include "bgpgen/cardinality.hpp"
#include "bgpgen/match_weights.hpp"
#include "bgpgen/matcher.hpp"
#include "bgpgen/nt_parser.hpp"
#include "bgpgen/sampler_complex.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <string>

using namespace bgpgen;

namespace {

struct Fixture {
  TermTable terms;
  TripleStore store;
  SubjectIndex fwd, rev;
};

std::unique_ptr<Fixture> load(const std::string& nt) {
  auto out = std::make_unique<Fixture>();
  std::vector<Triple> triples = parse_nt_buffer(nt.data(), nt.size(), out->terms, 0);
  std::vector<Triple>* sets[] = {&triples};
  out->terms.freeze(sets);
  out->store.build(std::move(triples), 1);
  out->fwd.build(out->store, out->terms.bnode_base());
  out->rev.build(out->store, out->terms.bnode_base(), true);
  return out;
}

std::string edge(const std::string& s, const std::string& p, const std::string& o) {
  return "<http://" + s + "> <http://" + p + "> <http://" + o + "> .\n";
}

/// The template as a query with every node and predicate a variable. Its
/// COUNT(*) is the number of homomorphisms, which is what the weights count
/// on an IRI-only graph when the template is a tree.
Query all_variable_query(const CompiledTemplate& t) {
  Query q;
  for (uint8_t s = 0; s < t.n_slots; ++s) q.var_names.push_back(t.slot_names[s]);
  for (size_t e = 0; e < t.triples.size(); ++e) {
    q.var_names.push_back("?p" + std::to_string(e));
    q.patterns.push_back({qvar(t.triples[e].s_slot), qvar(static_cast<uint32_t>(t.n_slots + e)),
                          qvar(t.triples[e].o_slot)});
  }
  return q;
}

// A small graph with parallel edges, a diamond and a tail.
const std::string kGraph = edge("a", "p", "b") + edge("a", "q", "b") + edge("a", "p", "c") +
                           edge("b", "p", "d") + edge("c", "p", "d") + edge("c", "q", "e") +
                           edge("d", "p", "f") + edge("e", "p", "f") + edge("x", "p", "a");

} // namespace

TEST_CASE("weight total equals the homomorphism count for tree templates") {
  auto g = load(kGraph);
  std::vector<CompiledTemplate> templates;
  templates.push_back(make_chain_template(1));
  templates.push_back(make_chain_template(2));
  templates.push_back(make_chain_template(3));
  templates.push_back(parse_template("?r B ?a, ?r B ?b, ?a B ?c"));      // out-tree
  templates.push_back(parse_template("?a B ?r, ?r B ?b"));               // one reverse step
  templates.push_back(parse_template("?x B ?m, ?y B ?m, ?m B ?z, ?z B ?w")); // in-star + tail
  for (const CompiledTemplate& t : templates) {
    MatchWeights w;
    w.compute(g->fwd, g->rev, t, g->terms.bnode_base());
    int64_t expected = count_query(g->store, g->fwd, g->rev, all_variable_query(t));
    REQUIRE(expected > 0);
    CHECK(w.total() == doctest::Approx(static_cast<double>(expected)));
  }
}

TEST_CASE("pruning removes dead-end starts on a tree template") {
  // One chain of length 4 plus 20 one-hop stubs that cannot start a 3-path.
  std::string nt = edge("a0", "p", "a1") + edge("a1", "p", "a2") + edge("a2", "p", "a3") +
                   edge("a3", "p", "a4");
  for (int i = 0; i < 20; ++i)
    nt += edge("s" + std::to_string(i), "q", "t" + std::to_string(i));
  auto g = load(nt);
  CompiledTemplate tmpl = make_chain_template(3);

  ComplexOptions co;
  co.num = 5;
  co.p_edge = 1.0;
  co.p_node = 1.0; // bind both ends, so the two 3-paths are distinct queries
  co.min_new_per_pass = 0;
  co.max_reshuffles = 2;

  SampleReport base, pruned;
  {
    Dedup dedup(DedupMethod::Wl);
    std::mt19937_64 rng(1);
    auto qs = sample_complex(g->store, g->fwd, g->rev, g->terms, tmpl, co, dedup, rng, &base);
    CHECK(qs.size() == 2);
  }
  co.prune = true;
  {
    Dedup dedup(DedupMethod::Wl);
    std::mt19937_64 rng(1);
    auto qs = sample_complex(g->store, g->fwd, g->rev, g->terms, tmpl, co, dedup, rng, &pruned);
    CHECK(qs.size() == 2);
  }
  CHECK(base.start_candidates == 24); // a0..a3 and the 20 stubs
  CHECK(base.failed_matches > 0);
  CHECK(pruned.start_candidates == 2); // only a0 and a1 can start a 3-path
  CHECK(pruned.failed_matches == 0);
  CHECK(pruned.tree_matches == doctest::Approx(2.0));
}

TEST_CASE("weighted sampling with root weights draws every match equally often") {
  // r1 has one 2-path below it, r2 has six (3 children x 2 grandchildren).
  std::string nt = edge("r1", "p", "c0") + edge("c0", "p", "g0");
  for (int c = 1; c <= 3; ++c) {
    nt += edge("r2", "p", "c" + std::to_string(c));
    for (int k = 0; k < 2; ++k)
      nt += edge("c" + std::to_string(c), "p", "g" + std::to_string(c) + std::to_string(k));
  }
  auto g = load(nt);
  CompiledTemplate tmpl = make_chain_template(2);
  MatchWeights w;
  w.compute(g->fwd, g->rev, tmpl, g->terms.bnode_base());
  REQUIRE(w.total() == doctest::Approx(7.0));

  std::vector<TermId> roots;
  std::vector<double> cum;
  double acc = 0.0;
  for (TermId v = 0; v < g->terms.bnode_base(); ++v) {
    double wv = w.weight(tmpl.root_slot, v);
    if (wv > 0.0) {
      acc += wv;
      roots.push_back(v);
      cum.push_back(acc);
    }
  }
  REQUIRE(roots.size() == 2);

  Matcher m(g->store, g->fwd, g->rev, tmpl, g->terms.bnode_base());
  m.set_weights(&w);
  std::mt19937_64 rng(7);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::map<std::vector<TermId>, int> seen;
  const int draws = 70000;
  for (int i = 0; i < draws; ++i) {
    size_t pos = static_cast<size_t>(std::upper_bound(cum.begin(), cum.end(), unit(rng) * acc) -
                                     cum.begin());
    REQUIRE(m.sample_weighted(roots[pos], rng)); // a tree: no rejection possible
    ++seen[m.node_assign()];
  }
  CHECK(seen.size() == 7);
  // Expected 10000 each; standard deviation ~93, so 500 is more than 5 sigma.
  for (const auto& [match, n] : seen) CHECK(std::abs(n - draws / 7) < 500);
}

TEST_CASE("early checks move the cycle-closing step forward and keep matches valid") {
  CompiledTemplate tmpl = parse_template("?a B ?b, ?b B ?c, ?c B ?a, ?a B ?x, ?a B ?y");
  auto check_pos = [](const CompiledTemplate& t) {
    for (size_t i = 0; i < t.steps.size(); ++i)
      if (t.steps[i].dir == Step::Dir::Check) return i;
    return t.steps.size();
  };
  CHECK(check_pos(tmpl) == 4); // BFS reaches b->c after both tails
  reorder_checks_early(tmpl);
  CHECK(check_pos(tmpl) == 2); // directly after b and c are assigned
  REQUIRE(tmpl.steps.size() == 5);

  // Every step still extends from an assigned slot.
  std::vector<bool> assigned(tmpl.n_slots, false);
  assigned[tmpl.root_slot] = true;
  for (const Step& st : tmpl.steps) {
    const TemplateTriple& tt = tmpl.triples[st.edge];
    if (st.dir == Step::Dir::Check) {
      CHECK(assigned[tt.s_slot]);
      CHECK(assigned[tt.o_slot]);
    } else {
      CHECK(assigned[st.dir == Step::Dir::Fwd ? tt.s_slot : tt.o_slot]);
      assigned[st.new_slot] = true;
    }
  }

  // One triangle with two tails, plus a fork that is not a triangle.
  auto g = load(edge("a", "p", "b") + edge("b", "p", "c") + edge("c", "p", "a") +
                edge("a", "q", "x") + edge("a", "q", "y") + edge("u", "p", "v") +
                edge("u", "p", "w"));
  ComplexOptions co;
  co.num = 10;
  co.prune = true;
  Dedup dedup(DedupMethod::Wl);
  std::mt19937_64 rng(3);
  auto qs = sample_complex(g->store, g->fwd, g->rev, g->terms, tmpl, co, dedup, rng);
  REQUIRE(!qs.empty());
  for (Query& q : qs) CHECK(count_query(g->store, g->fwd, g->rev, q) >= 1);
}

TEST_CASE("weighted sampling rejects draws whose closing edge is missing") {
  auto g = load(edge("a", "p", "b") + edge("b", "p", "c") + edge("a", "p", "c") +
                edge("x", "p", "y") + edge("x", "p", "z") + edge("y", "p", "w"));
  CompiledTemplate tmpl = parse_template("?a B ?b, ?b B ?c, ?a B ?c");
  ComplexOptions co;
  co.num = 10;
  co.sampling = ComplexOptions::Sampling::Weighted;
  co.max_attempts = 5000;
  co.max_consecutive = 5000;
  Dedup dedup(DedupMethod::Hash);
  std::mt19937_64 rng(5);
  SampleReport rep;
  auto qs = sample_complex(g->store, g->fwd, g->rev, g->terms, tmpl, co, dedup, rng, &rep);
  REQUIRE(qs.size() == 1); // the one triangle, deduplicated
  CHECK(rep.failed_matches > 0);
  CHECK(count_query(g->store, g->fwd, g->rev, qs[0]) >= 1);
}

TEST_CASE("auto_prune switches pruning on once failed attempts cost as much as the pass") {
  // One chain a0..a4 (two 3-paths) and 40 dead-end bushes: s -> 6 t -> 6 u
  // each. A 3-path attempt from a bush looks at ~80 candidates and fails.
  std::string nt = edge("a0", "p", "a1") + edge("a1", "p", "a2") + edge("a2", "p", "a3") +
                   edge("a3", "p", "a4");
  for (int b = 0; b < 40; ++b) {
    std::string s = "s" + std::to_string(b);
    for (int i = 0; i < 6; ++i) {
      std::string t = "t" + std::to_string(b) + "_" + std::to_string(i);
      nt += edge(s, "q", t);
      for (int j = 0; j < 6; ++j) nt += edge(t, "q", "u" + t + "_" + std::to_string(j));
    }
  }
  auto g = load(nt);
  CompiledTemplate tmpl = make_chain_template(3);
  ComplexOptions co;
  co.num = 5; // only two distinct queries exist, so the sampler keeps trying
  co.p_edge = 1.0;
  co.p_node = 1.0;
  co.min_new_per_pass = 0;

  SampleReport plain, automatic;
  {
    Dedup dedup(DedupMethod::Wl);
    std::mt19937_64 rng(11);
    auto qs = sample_complex(g->store, g->fwd, g->rev, g->terms, tmpl, co, dedup, rng, &plain);
    CHECK(qs.size() == 2);
  }
  co.auto_prune = true;
  {
    Dedup dedup(DedupMethod::Wl);
    std::mt19937_64 rng(11);
    auto qs = sample_complex(g->store, g->fwd, g->rev, g->terms, tmpl, co, dedup, rng, &automatic);
    CHECK(qs.size() == 2);
  }
  CHECK(!plain.auto_pruned);
  CHECK(automatic.auto_pruned);
  CHECK(automatic.start_candidates == 2); // a0 and a1
  CHECK(automatic.expansions < plain.expansions);
}

TEST_CASE("auto_prune stays off when attempts rarely fail") {
  auto g = load(kGraph);
  CompiledTemplate tmpl = make_chain_template(1);
  ComplexOptions co;
  co.num = 3;
  co.p_node = 1.0;
  co.auto_prune = true;
  co.min_new_per_pass = 0;
  Dedup dedup(DedupMethod::Wl);
  std::mt19937_64 rng(2);
  SampleReport rep;
  auto qs = sample_complex(g->store, g->fwd, g->rev, g->terms, tmpl, co, dedup, rng, &rep);
  CHECK(qs.size() == 3);
  CHECK(!rep.auto_pruned);
  CHECK(rep.weights_seconds == 0.0);
}

TEST_CASE("attempt budget bounds the search on a start node without a match") {
  // Hub h with 30 out-neighbours and no in-edge. "?r B ?a, ?r B ?b, ?z B ?r"
  // has no match, and the DFS tries every pair (a, b) before it knows.
  std::string nt;
  for (int i = 0; i < 30; ++i) nt += edge("h", "p", "t" + std::to_string(i));
  auto g = load(nt);
  CompiledTemplate tmpl = parse_template("?r B ?a, ?r B ?b, ?z B ?r");
  ComplexOptions co;
  co.num = 1;

  SampleReport plain, bounded;
  {
    Dedup dedup(DedupMethod::Wl);
    std::mt19937_64 rng(5);
    auto qs = sample_complex(g->store, g->fwd, g->rev, g->terms, tmpl, co, dedup, rng, &plain);
    CHECK(qs.empty());
  }
  co.attempt_budget = 50;
  {
    Dedup dedup(DedupMethod::Wl);
    std::mt19937_64 rng(5);
    auto qs = sample_complex(g->store, g->fwd, g->rev, g->terms, tmpl, co, dedup, rng, &bounded);
    CHECK(qs.empty());
  }
  CHECK(plain.attempts == 1);
  CHECK(plain.gave_up == 0);
  CHECK(plain.expansions == 30 + 30 * 30);
  CHECK(bounded.gave_up == 1);
  CHECK(bounded.expansions == 50);
}

TEST_CASE("an attempt budget that is never reached leaves the output unchanged") {
  auto g = load(kGraph);
  CompiledTemplate tmpl = make_chain_template(2);
  ComplexOptions co;
  co.num = 4;
  co.p_node = 1.0;
  co.min_new_per_pass = 0;
  std::vector<std::string> keys[2];
  for (int run = 0; run < 2; ++run) {
    co.attempt_budget = run == 0 ? 0 : 100000;
    Dedup dedup(DedupMethod::Wl);
    std::mt19937_64 rng(3);
    SampleReport rep;
    auto qs = sample_complex(g->store, g->fwd, g->rev, g->terms, tmpl, co, dedup, rng, &rep);
    CHECK(rep.gave_up == 0);
    for (const Query& q : qs) keys[run].push_back(canonical_bgp_key(q.patterns));
  }
  CHECK(!keys[0].empty());
  CHECK(keys[0] == keys[1]);
}

TEST_CASE("degree check rules out nodes with too few different neighbours and loses no match") {
  // h has three out-edges but only two different neighbours (like a in kGraph).
  // k has two out-neighbours and one in-neighbour, but only two neighbours in all.
  auto g = load(kGraph + edge("h", "p", "a") + edge("h", "q", "a") + edge("h", "p", "b") +
                edge("k", "p", "m") + edge("k", "p", "n") + edge("m", "q", "k"));
  TermId n = g->terms.bnode_base();
  const char* shapes[] = {
      "?r B ?a, ?r B ?b, ?r B ?c",
      "?r B ?a, ?r B ?b",
      "?x B ?y, ?y B ?z, ?y B ?w",
      "?a B ?m, ?b B ?m, ?m B ?c",
      "?a B ?b, ?b B ?c, ?a B ?c",
      "?z B ?r, ?r B ?a, ?r B ?b",
      "?a B ?b, ?b B ?a, ?b B ?c",
  };
  size_t ruled_out = 0, matches = 0;
  for (const char* shape : shapes) {
    CompiledTemplate tmpl = parse_template(shape);
    reorder_checks_early(tmpl);
    Matcher plain(g->store, g->fwd, g->rev, tmpl, n);
    Matcher checked(g->store, g->fwd, g->rev, tmpl, n);
    checked.set_degree_check(true);
    MatchWeights w;
    w.compute(g->fwd, g->rev, tmpl, n, /*degree_check=*/true);
    // Every row is shorter than the candidate cap, so match() finds a match
    // whenever one exists.
    for (TermId v = 0; v < n; ++v) {
      std::mt19937_64 r1(v), r2(v);
      bool found = plain.match(v, r1);
      bool found_checked = checked.can_root(v) && checked.match(v, r2);
      CHECK(found == found_checked);
      if (found) CHECK(w.weight(tmpl.root_slot, v) > 0.0);
      ruled_out += !checked.can_root(v) && g->fwd.out_degree(v) > 0;
      matches += found;
    }
  }
  CHECK(ruled_out > 0);
  CHECK(matches > 0);

  {
    // k passes the out- and the in-count but has only two neighbours in all.
    CompiledTemplate tmpl = parse_template("?z B ?r, ?r B ?a, ?r B ?b");
    Matcher checked(g->store, g->fwd, g->rev, tmpl, n);
    checked.set_degree_check(true);
    TermId k = g->terms.find_iri("http://k");
    REQUIRE(k != kInvalidTerm);
    CHECK(tmpl.slot_names[tmpl.root_slot] == "?r");
    CHECK(g->fwd.has_iri_neighbours(k, 2, n));
    CHECK(g->rev.has_iri_neighbours(k, 1, n));
    CHECK(!checked.can_root(k));
  }

  // No node has three different out-neighbours: with the check the sampler
  // does not start a single search.
  CompiledTemplate star = parse_template("?r B ?a, ?r B ?b, ?r B ?c");
  ComplexOptions co;
  co.num = 5;
  co.degree_check = true;
  Dedup dedup(DedupMethod::Symmetry);
  std::mt19937_64 rng(1);
  SampleReport rep;
  auto qs = sample_complex(g->store, g->fwd, g->rev, g->terms, star, co, dedup, rng, &rep);
  CHECK(qs.empty());
  CHECK(rep.attempts == 0);
  CHECK(rep.expansions == 0);
  CHECK(rep.skipped_starts > 0);
}

TEST_CASE("symmetry and wl give the same workload for the same seed") {
  auto g = load(kGraph);
  const char* shapes[] = {"?r B ?a, ?r B ?b", "?x B ?y, ?y B ?z, ?y B ?w",
                          "?r/B ?/B ?a/B, ?r/B ?/B ?b/B", "?x/B B ?y/B, ?y/B B ?z/B, ?y/B B ?w/B"};
  for (const char* shape : shapes) {
    CompiledTemplate tmpl = parse_template(shape);
    ComplexOptions co;
    co.num = 50;
    co.p_node = 0.5;
    co.p_edge = 0.5;
    std::vector<std::string> out[2];
    for (int run = 0; run < 2; ++run) {
      Dedup dedup(run == 0 ? DedupMethod::Wl : DedupMethod::Symmetry);
      std::mt19937_64 rng(7);
      for (const Query& q :
           sample_complex(g->store, g->fwd, g->rev, g->terms, tmpl, co, dedup, rng))
        out[run].push_back(render_sparql(q, g->terms));
    }
    CHECK(!out[0].empty());
    CHECK(out[0] == out[1]);
  }
}

TEST_CASE("predicate first chooses every predicate of a node equally often") {
  // h has 99 edges with predicate p and one with predicate q.
  std::string nt = edge("h", "q", "u");
  for (int i = 0; i < 99; ++i) nt += edge("h", "p", "t" + std::to_string(i));
  auto g = load(nt);
  CompiledTemplate tmpl = parse_template("?r B ?a");
  TermId h = g->terms.find_iri("http://h");
  TermId q = g->terms.find_iri("http://q");
  REQUIRE(h != kInvalidTerm);
  REQUIRE(q != kInvalidTerm);
  const PredIndex qi = g->store.pred_index(q);
  const size_t n_preds = g->store.predicate_count();

  auto share_of_q = [&](bool pred_first, double novelty, const uint32_t* uses) {
    Matcher m(g->store, g->fwd, g->rev, tmpl, g->terms.bnode_base());
    m.set_predicate_first(pred_first, novelty, uses, n_preds);
    std::mt19937_64 rng(1);
    int hits = 0;
    for (int i = 0; i < 4000; ++i) {
      REQUIRE(m.match(h, rng));
      hits += m.pred_assign()[0] == q;
      CHECK((m.pred_index_assign()[0] == qi) == (m.pred_assign()[0] == q));
    }
    return hits / 4000.0;
  };
  CHECK(share_of_q(false, 0.0, nullptr) < 0.03);
  double uniform = share_of_q(true, 0.0, nullptr);
  CHECK(uniform > 0.45);
  CHECK(uniform < 0.55);

  // novelty 1: weight 1 / (1 + uses). p used 9 times, q never: q gets 10/11.
  std::vector<uint32_t> uses(n_preds, 9);
  uses[qi] = 0;
  double novel = share_of_q(true, 1.0, uses.data());
  CHECK(novel > 0.87);
  CHECK(novel < 0.95);
}

TEST_CASE("skip_known finds every query of a small graph and none twice") {
  auto g = load(kGraph + edge("h", "p", "a") + edge("h", "q", "a") + edge("h", "p", "b") +
                edge("k", "p", "m") + edge("k", "p", "n") + edge("m", "q", "k"));
  struct Case {
    const char* shape;
    double p_node;
  };
  for (Case c : {Case{"?r B ?a, ?r B ?b", 0.0}, Case{"?r B ?a, ?a B ?b, ?z B ?r", 0.0},
                 Case{"?r B ?a, ?r B ?b, ?b B ?c", 0.0}, Case{"?r/B B ?a, ?a B ?b/B", 1.0}}) {
    CAPTURE(c.shape);
    CompiledTemplate tmpl = parse_template(c.shape);
    auto run = [&](uint64_t seed, bool skip_known, bool pred_first, double novelty,
                   size_t per_seed = 1, bool retire = false, unsigned threads = 1) {
      ComplexOptions co;
      co.num = 1000;
      co.p_node = c.p_node;
      co.min_new_per_pass = 0;
      co.skip_known = skip_known;
      co.pred_first = pred_first;
      co.novelty = novelty;
      co.per_seed = per_seed;
      co.retire_seeds = retire;
      co.threads = threads;
      Dedup dedup(DedupMethod::Symmetry);
      std::mt19937_64 rng(seed);
      auto qs = sample_complex(g->store, g->fwd, g->rev, g->terms, tmpl, co, dedup, rng);
      std::vector<std::string> keys;
      for (const Query& q : qs) {
        CHECK(count_query(g->store, g->fwd, g->rev, q) >= 1);
        keys.push_back(canonical_bgp_key(q.patterns));
      }
      std::sort(keys.begin(), keys.end());
      CHECK(std::adjacent_find(keys.begin(), keys.end()) == keys.end());
      return keys;
    };
    // The search from a start node is complete here (no budget, no node with
    // more than 50 neighbours), so every seed ends with the same set.
    std::vector<std::string> all = run(1, true, false, 0.0);
    CHECK(!all.empty());
    CHECK(run(2, true, false, 0.0) == all);
    // Several queries per visit, and leaving out start nodes that gave no new
    // query, lose nothing when the search is complete.
    for (size_t per_seed : {size_t{0}, size_t{1}, size_t{4}})
      for (bool retire : {false, true}) CHECK(run(3, true, false, 0.0, per_seed, retire) == all);
    // Several threads find the same set, and the same workload for the same
    // seed and number of threads.
    for (unsigned threads : {2u, 3u, 8u}) {
      CAPTURE(threads);
      CHECK(run(4, true, false, 0.0, 1, false, threads) == all);
      CHECK(run(4, true, true, 1.0, 0, true, threads) == run(4, true, true, 1.0, 0, true, threads));
    }
    for (uint64_t seed : {1, 2, 3}) {
      std::vector<std::string> plain = run(seed, false, false, 0.0);
      CHECK(std::includes(all.begin(), all.end(), plain.begin(), plain.end()));
      for (double novelty : {0.0, 1.0}) {
        std::vector<std::string> guided = run(seed, true, true, novelty);
        CHECK(std::includes(all.begin(), all.end(), guided.begin(), guided.end()));
        CHECK(guided.size() >= plain.size());
      }
    }
  }
}

TEST_CASE("seed groups visit one start node of every predicate set first") {
  // 60 nodes with out-predicate p, one node with out-predicate q.
  std::string nt = edge("b", "q", "y");
  for (int i = 0; i < 60; ++i) nt += edge("a" + std::to_string(i), "p", "x");
  auto g = load(nt);
  CompiledTemplate tmpl = parse_template("?r B ?a");
  for (uint64_t seed : {1, 2, 3, 4, 5}) {
    ComplexOptions co;
    co.num = 2;
    co.skip_known = true;
    co.seed_groups = true;
    Dedup dedup(DedupMethod::Symmetry);
    std::mt19937_64 rng(seed);
    SampleReport rep;
    auto qs = sample_complex(g->store, g->fwd, g->rev, g->terms, tmpl, co, dedup, rng, &rep);
    CHECK(qs.size() == 2);
    CHECK(rep.attempts == 2); // one node of each of the two sets
  }
}

TEST_CASE("cardinality targets fill every range with counted queries") {
  // One predicate per result size: "?a B ?b" with predicate p<k> has k results.
  std::string nt;
  for (int k : {2, 5, 8, 20, 50, 80})
    for (int i = 0; i < k; ++i)
      nt += edge("s" + std::to_string(k) + "_" + std::to_string(i), "p" + std::to_string(k),
                 "o" + std::to_string(i));
  auto g = load(nt);
  CompiledTemplate tmpl = parse_template("?a B ?b");
  for (int variant = 0; variant < 4; ++variant) {
    const bool steer = variant & 1;
    CAPTURE(variant);
    ComplexOptions co;
    co.threads = variant & 2 ? 3 : 1;
    co.card.buckets = 2;
    co.card.max_card = 100; // ranges [1, 10) and [10, 100)
    co.card.steer = steer;
    {
      co.num = 4;
      Dedup dedup(DedupMethod::Symmetry);
      std::mt19937_64 rng(1);
      SampleReport rep;
      auto qs = sample_complex(g->store, g->fwd, g->rev, g->terms, tmpl, co, dedup, rng, &rep);
      REQUIRE(qs.size() == 4);
      CHECK(rep.bucket_fill == std::vector<size_t>{2, 2});
      for (const Query& q : qs) CHECK(q.y == count_query(g->store, g->fwd, g->rev, q));
      CHECK(rep.stop_reason.empty());
    }
    {
      co.num = 10; // five per range asked, three exist
      co.min_new_per_pass = 0;
      Dedup dedup(DedupMethod::Symmetry);
      std::mt19937_64 rng(1);
      SampleReport rep;
      auto qs = sample_complex(g->store, g->fwd, g->rev, g->terms, tmpl, co, dedup, rng, &rep);
      CHECK(qs.size() == 6);
      CHECK(rep.bucket_fill == std::vector<size_t>{3, 3});
      CHECK(!rep.stop_reason.empty());
    }
    {
      co.num = 4;
      co.card.max_card = 10; // one range would do; two ranges: [1, 3.16) and [3.16, 10)
      Dedup dedup(DedupMethod::Symmetry);
      std::mt19937_64 rng(1);
      SampleReport rep;
      auto qs = sample_complex(g->store, g->fwd, g->rev, g->terms, tmpl, co, dedup, rng, &rep);
      CHECK(rep.bucket_fill == std::vector<size_t>{1, 2}); // 2 | 5, 8
      // 20, 50, 80; threads of one round may each count and drop them.
      if (co.threads == 1)
        CHECK(rep.dropped_outside == 3);
      else
        CHECK(rep.dropped_outside >= 3);
      for (const Query& q : qs) CHECK(q.y < 10);
    }
  }
}
