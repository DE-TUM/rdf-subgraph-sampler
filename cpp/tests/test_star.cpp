#include "doctest.h"

#include "bgpgen/nt_parser.hpp"
#include "bgpgen/sampler_star.hpp"

#include <memory>
#include <string>

using namespace bgpgen;

namespace {

struct Fixture {
  TermTable terms;
  TripleStore store;
  SubjectIndex index;
};

std::unique_ptr<Fixture> load(const std::string& nt) {
  auto out = std::make_unique<Fixture>();
  std::vector<Triple> triples = parse_nt_buffer(nt.data(), nt.size(), out->terms, 0);
  std::vector<Triple>* sets[] = {&triples};
  out->terms.freeze(sets);
  out->store.build(std::move(triples), 1);
  out->index.build(out->store, out->terms.bnode_base());
  return out;
}

const char* kToyGraph = "<http://s1> <http://p1> <http://a> .\n"
                        "<http://s1> <http://p2> \"v1\" .\n"
                        "<http://s1> <http://p3> <http://b> .\n"
                        "<http://s2> <http://p1> <http://a> .\n"
                        "<http://s2> <http://p2> \"v2\" .\n"
                        "<http://s2> <http://p3> <http://c> .\n"
                        "<http://s3> <http://p1> <http://a> .\n";

} // namespace

TEST_CASE("star sampler invariants on a toy graph") {
  auto g = load(kToyGraph);
  StarOptions opt;
  opt.size = 2;
  opt.num = 100; // more than exist; sampler must stop on its own
  opt.p_predicate = 1.0;
  opt.min_objects = 0;
  opt.max_objects = 1;

  Dedup dedup(DedupMethod::Hash);
  std::mt19937_64 rng(42);
  SampleReport rep;
  std::vector<StarRecord> recs = sample_stars(g->store, g->index, g->terms, opt, dedup, rng, &rep);

  CHECK(!recs.empty());
  CHECK(!rep.stop_reason.empty()); // toy graph can't yield 100 unique queries
  for (const StarRecord& r : recs) {
    REQUIRE(r.query.patterns.size() == 2);
    // Subject is always variable 0 = "?s"; star roots only s1/s2 (degree >= 2).
    std::string_view subj = g->terms.lexical(r.meta.source_subject);
    CHECK((subj == "http://s1" || subj == "http://s2"));
    for (const TriplePattern& tp : r.query.patterns) {
      CHECK(tp.s == qvar(0));
      CHECK(r.query.var_names[0] == "?s");
      // Every bound (pred, obj) pair must exist on the source subject.
      if (!q_is_var(tp.p) && !q_is_var(tp.o)) {
        PredIndex p = g->store.pred_index(tp.p);
        CHECK(g->store.has(r.meta.source_subject, p, tp.o));
      }
    }
    CHECK(r.meta.star_size >= 2);
    CHECK(r.meta.instantiated_objects.size() == r.meta.n_objects_instantiated);
  }
}

TEST_CASE("hash dedup collapses same pattern from different subjects") {
  auto g = load(kToyGraph);
  // s1 and s2 share the identical (p1, <a>) edge; with all objects free the
  // queries {?s p1 ?o0, ?s p2 ?o1} from s1 and s2 are duplicates.
  StarOptions opt;
  opt.size = 3;
  opt.num = 100;
  opt.p_predicate = 1.0;
  opt.min_objects = 0;
  opt.max_objects = 0; // never instantiate objects

  Dedup dedup(DedupMethod::Hash);
  std::mt19937_64 rng(7);
  SampleReport rep;
  std::vector<StarRecord> recs = sample_stars(g->store, g->index, g->terms, opt, dedup, rng, &rep);
  // Both size-3 stars (s1, s2) have predicate multiset {p1,p2,p3} and free
  // objects -> exactly one unique query survives.
  CHECK(recs.size() == 1);
  CHECK(rep.duplicates_skipped > 0);
}
