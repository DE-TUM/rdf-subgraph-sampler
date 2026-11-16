#include "doctest.h"

#include "bgpgen/graph.hpp"
#include "bgpgen/nt_parser.hpp"
#include "bgpgen/term_table.hpp"

#include <memory>
#include <string>

using namespace bgpgen;

namespace {

// TermTable is non-movable (arena self-pointers), so tests hold it on the heap.
struct Loaded {
  TermTable terms;
  TripleStore store;
};

std::unique_ptr<Loaded> load(const std::string& nt) {
  auto out = std::make_unique<Loaded>();
  std::vector<Triple> triples = parse_nt_buffer(nt.data(), nt.size(), out->terms, 0);
  std::vector<Triple>* sets[] = {&triples};
  out->terms.freeze(sets);
  out->store.build(std::move(triples), 1);
  return out;
}

} // namespace

TEST_CASE("parse interns IRIs bracket-less and unifies positions") {
  auto g = load("<http://a> <http://p> <http://a> .\n");
  CHECK(g->store.triple_count() == 1);
  TermId a = g->terms.find_iri("http://a");
  REQUIRE(a != kInvalidTerm);
  PredIndex p = g->store.pred_index(g->terms.find_iri("http://p"));
  REQUIRE(p != kNoPred);
  // Same IRI in subject and object position is one term.
  IdSpan objs = g->store.objects_of(a, p);
  REQUIRE(objs.size() == 1);
  CHECK(objs[0] == a);
}

TEST_CASE("literals are interned as raw tokens, verbatim") {
  auto g = load("<http://a> <http://p> \"hi \\\"x\\\"\"@en .\n"
                  "<http://a> <http://q> \"5\"^^<http://www.w3.org/2001/XMLSchema#int> .\n"
                  "<http://a> <http://q> \"plain\" .\n");
  CHECK(g->store.triple_count() == 3);
  TermId a = g->terms.find_iri("http://a");
  PredIndex p = g->store.pred_index(g->terms.find_iri("http://p"));
  PredIndex q = g->store.pred_index(g->terms.find_iri("http://q"));
  REQUIRE(g->store.objects_of(a, p).size() == 1);
  CHECK(g->terms.lexical(g->store.objects_of(a, p)[0]) == "\"hi \\\"x\\\"\"@en");
  IdSpan q_objs = g->store.objects_of(a, q);
  REQUIRE(q_objs.size() == 2);
  bool saw_typed = false, saw_plain = false;
  for (TermId o : q_objs) {
    std::string_view lex = g->terms.lexical(o);
    if (lex == "\"5\"^^<http://www.w3.org/2001/XMLSchema#int>") saw_typed = true;
    if (lex == "\"plain\"") saw_plain = true;
    CHECK(g->terms.is_literal(o));
  }
  CHECK(saw_typed);
  CHECK(saw_plain);
}

TEST_CASE("duplicate triples are removed (set semantics)") {
  auto g = load("<http://a> <http://p> <http://b> .\n"
                  "<http://a> <http://p> <http://b> .\n"
                  "<http://a> <http://p> <http://c> .\n");
  CHECK(g->store.triple_count() == 2);
}

TEST_CASE("forward and reverse CSR agree") {
  auto g = load("<http://a> <http://p> <http://b> .\n"
                  "<http://c> <http://p> <http://b> .\n"
                  "<http://a> <http://p> <http://d> .\n");
  TermId a = g->terms.find_iri("http://a");
  TermId b = g->terms.find_iri("http://b");
  TermId c = g->terms.find_iri("http://c");
  PredIndex p = g->store.pred_index(g->terms.find_iri("http://p"));
  CHECK(g->store.objects_of(a, p).size() == 2);
  IdSpan subs = g->store.subjects_of(b, p);
  REQUIRE(subs.size() == 2);
  CHECK(contains(subs, a));
  CHECK(contains(subs, c));
  CHECK(g->store.has(a, p, b));
  CHECK(!g->store.has(c, p, g->terms.find_iri("http://d")));
}

TEST_CASE("blank nodes and comments parse") {
  auto g = load("# a comment\n"
                  "_:x <http://p> _:y .\n"
                  "_:x <http://p> \"lit\" .\n");
  CHECK(g->store.triple_count() == 2);
  CHECK(g->terms.size() == 4); // 1 iri (p), 2 bnodes, 1 literal
}

TEST_CASE("term counts segment correctly") {
  auto g = load("_:x <http://p> _:y .\n"
                  "<http://s> <http://p> \"lit\" .\n");
  // iris: http://p, http://s; bnodes: x, y; literals: "lit"
  CHECK(g->terms.iri_count() == 2);
  CHECK(g->terms.bnode_base() == 2);
  CHECK(g->terms.literal_base() == 4);
  CHECK(g->terms.size() == 5);
}
