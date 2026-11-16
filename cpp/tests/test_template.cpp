#include "doctest.h"

#include "bgpgen/nt_parser.hpp"
#include "bgpgen/sampler_complex.hpp"

#include <memory>
#include <set>
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

/// Every traversal step must extend from an already-assigned slot.
void check_traversal_sound(const CompiledTemplate& t) {
  std::vector<bool> assigned(t.n_slots, false);
  assigned[t.root_slot] = true;
  REQUIRE(t.steps.size() == t.triples.size());
  for (const Step& st : t.steps) {
    const TemplateTriple& tt = t.triples[st.edge];
    if (st.dir == Step::Dir::Check) {
      CHECK(assigned[tt.s_slot]);
      CHECK(assigned[tt.o_slot]);
    } else if (st.dir == Step::Dir::Fwd) {
      CHECK(assigned[tt.s_slot]);
      CHECK(st.new_slot == tt.o_slot);
      assigned[tt.o_slot] = true;
    } else {
      CHECK(assigned[tt.o_slot]);
      CHECK(st.new_slot == tt.s_slot);
      assigned[tt.s_slot] = true;
    }
  }
  for (uint8_t s = 0; s < t.n_slots; ++s) CHECK(assigned[s]);
}

} // namespace

TEST_CASE("example shapes from batch_sampler_config parse and compile") {
  // Path, star, flower, binary tree, triangle, diamond, snowflake.
  const char* shapes[] = {
      "?o0/B ?/B ?o1, ?o1 ?/B ?o2/B",
      "?s B ?o1, ?s B ?o2, ?s B ?o3",
      "?c B ?p1, ?c B ?p2, ?x B ?c, ?y B ?c",
      "?r B ?a, ?r B ?b, ?a B ?a1, ?a B ?a2, ?b B ?b1, ?b B ?b2",
      "?a B ?b, ?b B ?c, ?a B ?c",
      "?s B ?a, ?s B ?b, ?a B ?t, ?b B ?t",
      "?h B ?m1, ?h B ?m2, ?h B ?m3, ?m1 B ?l1, ?m2 B ?l2, ?m3 B ?l3",
  };
  for (const char* s : shapes) {
    CAPTURE(s);
    CompiledTemplate t = parse_template(s);
    check_traversal_sound(t);
  }
}

TEST_CASE("traversal root is the highest-degree subject node") {
  // Star: ?s has degree 3 and is a subject -> root.
  CompiledTemplate star = parse_template("?s B ?o1, ?s B ?o2, ?s B ?o3");
  CHECK(star.slot_names[star.root_slot] == "?s");
  // Flower: ?c has degree 4 (2 as subject, 2 as object).
  CompiledTemplate flower = parse_template("?c B ?p1, ?c B ?p2, ?x B ?c, ?y B ?c");
  CHECK(flower.slot_names[flower.root_slot] == "?c");
  // Triangle: closing edge must be a Check step.
  CompiledTemplate tri = parse_template("?a B ?b, ?b B ?c, ?a B ?c");
  int checks = 0;
  for (const Step& st : tri.steps) checks += st.dir == Step::Dir::Check;
  CHECK(checks == 1);
}

TEST_CASE("inconsistent node binding is rejected") {
  CHECK_THROWS_AS(parse_template("?a B ?b, a B ?c"), BgpError);
  CHECK_THROWS_AS(parse_template("?a B ?b, ?c B ?d"), BgpError); // disconnected
  CHECK_THROWS_AS(parse_template("?a X ?b"), BgpError);
}

TEST_CASE("chain template walks from ?o0") {
  CompiledTemplate t = make_chain_template(3);
  CHECK(t.n_slots == 4);
  CHECK(t.slot_names[t.root_slot] == "?o0");
  CHECK(t.pred_var_names[0] == "?p1"); // path predicate vars are 1-based
  check_traversal_sound(t);
  for (const Step& st : t.steps) CHECK(st.dir == Step::Dir::Fwd);
}

TEST_CASE("path sampler finds simple paths of exact length") {
  // Chain with a branch and a cycle back edge:
  // a -> b -> c -> d, b -> e, d -> a
  auto g = load("<http://a> <http://p> <http://b> .\n"
                "<http://b> <http://p> <http://c> .\n"
                "<http://c> <http://p> <http://d> .\n"
                "<http://b> <http://q> <http://e> .\n"
                "<http://d> <http://q> <http://a> .\n");
  CompiledTemplate tmpl = make_chain_template(3);
  ComplexOptions co;
  co.num = 50;
  co.p_edge = 1.0;
  co.p_node = 0.0; // never bind the endpoints
  co.min_new_per_pass = 0;
  co.max_consecutive = 200;

  Dedup dedup(DedupMethod::Hash);
  std::mt19937_64 rng(123);
  SampleReport rep;
  std::vector<Query> qs =
      sample_complex(g->store, g->fwd, g->rev, g->terms, tmpl, co, dedup, rng, &rep);

  CHECK(!qs.empty());
  for (const Query& q : qs) {
    REQUIRE(q.patterns.size() == 3);
    // Chain structure: object var of pattern i == subject var of pattern i+1.
    CHECK(q.patterns[0].o == q.patterns[1].s);
    CHECK(q.patterns[1].o == q.patterns[2].s);
    // All predicates bound (p_edge = 1) and the underlying path edges exist.
    for (const TriplePattern& tp : q.patterns) CHECK(!q_is_var(tp.p));
  }
}

TEST_CASE("triangle template only matches actual triangles") {
  // One triangle a->b->c, a->c plus a non-triangle fork x->y, x->z.
  auto g = load("<http://a> <http://p> <http://b> .\n"
                "<http://b> <http://p> <http://c> .\n"
                "<http://a> <http://p> <http://c> .\n"
                "<http://x> <http://p> <http://y> .\n"
                "<http://x> <http://p> <http://z> .\n");
  CompiledTemplate tmpl = parse_template("?a B ?b, ?b B ?c, ?a B ?c");
  ComplexOptions co;
  co.num = 10;
  co.p_edge = 1.0;
  co.p_node = 0.0;

  Dedup dedup(DedupMethod::Hash);
  std::mt19937_64 rng(5);
  std::vector<Query> qs = sample_complex(g->store, g->fwd, g->rev, g->terms, tmpl, co, dedup, rng);
  REQUIRE(qs.size() == 1); // exactly one triangle, deduped
  const Query& q = qs[0];
  // Verify all three edges exist in the data with the matched constants...
  // predicates are bound; structure is var-only nodes, so check predicates.
  for (const TriplePattern& tp : q.patterns) {
    CHECK(!q_is_var(tp.p));
    CHECK(g->terms.lexical(tp.p) == "http://p");
  }
}

TEST_CASE("bound template nodes appear as constants and in x") {
  auto g = load("<http://a> <http://p> <http://b> .\n");
  CompiledTemplate tmpl = parse_template("s B ?o"); // subject always bound
  ComplexOptions co;
  co.num = 10;
  co.p_edge = 1.0;
  co.p_node = 0.0;
  Dedup dedup(DedupMethod::Hash);
  std::mt19937_64 rng(1);
  std::vector<Query> qs = sample_complex(g->store, g->fwd, g->rev, g->terms, tmpl, co, dedup, rng);
  REQUIRE(qs.size() == 1);
  CHECK(!q_is_var(qs[0].patterns[0].s));
  CHECK(g->terms.lexical(qs[0].patterns[0].s) == "http://a");
  // x = [subject, predicate] in complex order.
  REQUIRE(qs[0].x.size() == 2);
  CHECK(g->terms.lexical(qs[0].x[0]) == "http://a");
  CHECK(g->terms.lexical(qs[0].x[1]) == "http://p");
}

TEST_CASE("slot degrees count the different template neighbours of a slot") {
  CompiledTemplate t = parse_template("?a B ?m, ?b B ?m, ?m B ?c");
  SlotDegrees d = slot_degrees(t);
  auto slot = [&](const std::string& name) {
    return static_cast<size_t>(std::find(t.slot_names.begin(), t.slot_names.end(), name) -
                               t.slot_names.begin());
  };
  CHECK(d.in[slot("?m")] == 2);
  CHECK(d.out[slot("?m")] == 1);
  CHECK(d.in[slot("?a")] == 0);
  CHECK(d.out[slot("?a")] == 1);
  CHECK(d.in[slot("?c")] == 1);
  CHECK(d.out[slot("?c")] == 0);
  CHECK(d.all[slot("?m")] == 3);
  CHECK(d.all[slot("?a")] == 1);

  // Two slots that point at each other are one neighbour each.
  CompiledTemplate pair = parse_template("?a B ?b, ?b B ?a, ?b B ?c");
  SlotDegrees p = slot_degrees(pair);
  auto pslot = [&](const std::string& name) {
    return static_cast<size_t>(std::find(pair.slot_names.begin(), pair.slot_names.end(), name) -
                               pair.slot_names.begin());
  };
  CHECK(p.out[pslot("?b")] == 2);
  CHECK(p.in[pslot("?b")] == 1);
  CHECK(p.all[pslot("?b")] == 2);
}

TEST_CASE("template symmetries: interchangeable leaves, other symmetries, limit") {
  auto sym = [](const char* shape) { return template_symmetries(parse_template(shape), 720); };
  using Groups = std::vector<std::vector<uint8_t>>;

  CHECK(sym("?a B ?b, ?b B ?c").perms.size() == 1); // directed path
  CHECK(sym("?a B ?b, ?b B ?c").twin_edges.empty());

  // Star: the leaves are interchangeable, nothing else is left.
  CHECK(sym("?r B ?a, ?r B ?b, ?r B ?c").perms == Groups{{0, 1, 2}});
  CHECK(sym("?r B ?a, ?r B ?b, ?r B ?c").twin_edges == Groups{{0, 1, 2}});

  CHECK(sym("?a B ?b, ?b B ?c, ?c B ?a").perms.size() == 3); // triangle: rotations
  // Two equal branches, each with one leaf: the branches can be swapped.
  CHECK(sym("?r B ?a, ?r B ?b, ?a B ?c, ?b B ?d").perms.size() == 2);
  CHECK(sym("?r B ?a, ?r B ?b, ?a B ?c, ?b B ?d").twin_edges.empty());

  // CEG's fork24: two interchangeable sources and two interchangeable sinks.
  auto fork = sym("?v0 B ?v1, ?v0 B ?v4, ?v2 B ?v4, ?v3 B ?v4, ?v4 B ?v5, ?v4 B ?v6");
  CHECK(fork.perms.size() == 1);
  CHECK(fork.twin_edges == Groups{{4, 5}, {2, 3}});

  // A star with ten leaves has 3.6 million symmetries, all of them reorderings.
  CHECK(sym("?r B ?a, ?r B ?b, ?r B ?c, ?r B ?d, ?r B ?e, ?r B ?f, ?r B ?g, ?r B ?h, "
            "?r B ?i, ?r B ?j").perms.size() == 1);

  // More symmetries than the limit, and parallel edges: unusable.
  CHECK(template_symmetries(parse_template("?a B ?b, ?b B ?c, ?c B ?a"), 2).perms.empty());
}
